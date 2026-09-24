//! The draw of a token as one kernel, for the CubeCL backends.
//!
//! As tensor operations, a draw is eight kernels, two of them over a vocabulary-squared
//! matrix, see [`Sampling::draw`](crate::qwen3::sampling::Sampling::draw). Here it is one kernel run
//! by a single cube: the logits go to shared memory, the top-k threshold comes out of a radix
//! select over the histogram of their ordered bits, the nucleus threshold out of the same
//! select over their probability mass, and the draw is the largest of the kept logits plus
//! Gumbel noise. A few microseconds where the operations took fifty.
//!
//! The kernel reaches the tensors through a backend extension: the trait below is implemented
//! for the CubeCL backend, which launches the kernel, and for the fusion backend wrapping it,
//! which registers the draw as a custom operation of its stream, so that it takes its place
//! among the fused operations around it and gets recorded by a graph capture like them.

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Dispatch, DispatchDevice, backend_extension};
use burn::tensor::{DType, Device, Int, Shape, Tensor as BurnTensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use burn_fusion::stream::{Operation, StreamId};
use burn_fusion::{ExecutionError, Fusion, FusionBackend, FusionRuntime};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, ScalarIr, TensorIr};
use cubecl::{CubeCount, CubeDim, RuntimeId, prelude::*};
use std::marker::PhantomData;

/// Sampling of a token on the device, see the module documentation.
#[backend_extension(Cube: cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "cpu"
)))]
pub trait SamplingBackend: burn::backend::Backend {
    /// Draws a token out of `logits` (vocab,) divided by `temperature`, with `noise` (vocab,)
    /// uniform in (0, 1): among the `top_k` largest (0 for all of them) whose nucleus is
    /// `top_p` (1 for all of them), or the largest one when `greedy`. Returns the index of the
    /// token as a one-element tensor.
    fn draw_token(
        logits: FloatTensor<Self>,
        noise: FloatTensor<Self>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        greedy: bool,
    ) -> IntTensor<Self>;
}

/// Whether [`draw_token`] runs on `device`: the kernel is written for a GPU, one cube of many
/// units sharing memory.
pub fn available(device: &Device) -> bool {
    matches!(device.as_dispatch(), DispatchDevice::Cube(device) if device.runtime() != RuntimeId::Cpu)
}

/// [`SamplingBackend::draw_token`] on tensors.
pub fn draw_token(
    logits: BurnTensor<1>,
    noise: BurnTensor<1>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    greedy: bool,
) -> BurnTensor<1, Int> {
    BurnTensor::from_dispatch(<Dispatch as SamplingBackend>::draw_token(
        logits.into_dispatch(),
        noise.into_dispatch(),
        temperature,
        top_k,
        top_p,
        greedy,
    ))
}

impl SamplingBackend for CubeBackend {
    fn draw_token(
        logits: FloatTensor<Self>,
        noise: FloatTensor<Self>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        greedy: bool,
    ) -> IntTensor<Self> {
        assert_eq!(logits.dtype, DType::F32, "the logits are sampled in f32");
        assert_eq!(noise.dtype, DType::F32, "the noise is f32");
        let logits = into_contiguous(logits);
        let noise = into_contiguous(noise);
        let vocab = logits.meta.shape()[0];
        let client = logits.client.clone();
        let output = CubeTensor::new_contiguous(
            client.clone(),
            logits.device.clone(),
            Shape::new([1]),
            client.empty(size_of::<i32>()),
            DType::I32,
        );
        // As many units as a cube holds on the CUDA-like runtimes, the portable maximum
        // elsewhere.
        let units = match logits.device.runtime() {
            RuntimeId::Cuda | RuntimeId::Hip => 1024,
            _ => 256,
        };
        draw_token_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(units),
            logits.into_tensor_arg(),
            noise.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            temperature,
            top_k as u32,
            top_p,
            u32::from(greedy),
            vocab,
        );
        output
    }
}

/// The draw as a custom operation of the fusion stream, executed on the inner backend.
struct DrawToken<B> {
    desc: CustomOpIr,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    greedy: bool,
    backend: PhantomData<B>,
}

impl<B> core::fmt::Debug for DrawToken<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DrawToken")
            .field("desc", &self.desc)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("greedy", &self.greedy)
            .finish()
    }
}

impl<B: FusionBackend + SamplingBackend> Operation<B::FusionRuntime> for DrawToken<B> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([logits, noise], [out]) = self.desc.as_fixed();
        let logits = handles.get_float_tensor::<B>(logits);
        let noise = handles.get_float_tensor::<B>(noise);
        let output = B::draw_token(
            logits,
            noise,
            self.temperature,
            self.top_k,
            self.top_p,
            self.greedy,
        );
        handles.register_int_tensor::<B>(&out.id, output);
        Ok(())
    }
}

impl<B: FusionBackend + SamplingBackend> SamplingBackend for Fusion<B> {
    fn draw_token(
        logits: FloatTensor<Self>,
        noise: FloatTensor<Self>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        greedy: bool,
    ) -> IntTensor<Self> {
        let client = logits.client.clone();
        let streams = StreamId::current();
        let out = TensorIr::uninit(client.create_empty_handle(), Shape::new([1]), DType::I32);
        // The scalars are part of the description so that draws with different settings never
        // share a cached plan.
        let desc = CustomOpIr::with_scalars(
            "draw_token",
            &[logits.into_ir(), noise.into_ir()],
            &[out],
            vec![
                ScalarIr::Float(temperature as f64),
                ScalarIr::UInt(top_k as u64),
                ScalarIr::Float(top_p as f64),
                ScalarIr::Bool(greedy),
            ],
        );
        let op = DrawToken::<B> {
            desc: desc.clone(),
            temperature,
            top_k,
            top_p,
            greedy,
            backend: PhantomData,
        };
        client
            .register(streams, OperationIr::Custom(desc), op)
            .output()
    }
}

/// The bits of `x` as an unsigned integer that orders like `x`.
#[cube]
fn ordered_key(x: f32) -> u32 {
    let bits = u32::reinterpret(x);
    select(bits >> 31 != 0, !bits, bits | 0x8000_0000u32)
}

/// The key of the `rank`-th largest of the `vals` (`rank` counts from 1) by a radix select:
/// every round histograms the next eight bits of the keys that share the prefix found so far,
/// and the first plane, scanning the histogram from the top, finds the bin the rank falls in.
/// Every unit of the cube takes part. `found` is scratch for the two numbers the scan passes on.
#[cube]
fn rank_key(
    vals: &Shared<[f32]>,
    hist: &mut Shared<[Atomic<u32>]>,
    found: &mut Shared<[u32]>,
    rank: u32,
    #[comptime] vocab: usize,
) -> u32 {
    let unit = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    let mut prefix = 0u32.runtime();
    let mut mask = 0u32.runtime();
    let mut remaining = rank;
    for round in 0..4u32 {
        let shift = 24u32 - 8u32 * round;
        for b in range_stepped(unit, 256usize, units) {
            hist[b].store(0u32);
        }
        sync_cube();
        for i in range_stepped(unit, vocab, units) {
            let key = ordered_key(vals[i]);
            if key & mask == prefix {
                hist[((key >> shift) & 0xFFu32) as usize].fetch_add(1u32);
            }
        }
        sync_cube();
        if PLANE_POS == 0 {
            // Every lane owns a run of bins, from the top down.
            let per = 256u32 / PLANE_DIM;
            let top = 255u32 - UNIT_POS_PLANE * per;
            let mut local = 0u32.runtime();
            for j in 0..per {
                local += hist[(top - j) as usize].load();
            }
            let above = plane_exclusive_sum(local);
            if above < remaining && remaining <= above + local {
                let mut cum = above;
                let mut done = false.runtime();
                for j in 0..per {
                    if !done {
                        let bin = top - j;
                        let count = hist[bin as usize].load();
                        cum += count;
                        if cum >= remaining {
                            found[0] = bin;
                            found[1] = remaining - (cum - count);
                            done = true;
                        }
                    }
                }
            }
        }
        sync_cube();
        prefix |= found[0] << shift;
        remaining = found[1];
        mask |= 0xFFu32 << shift;
        sync_cube();
    }
    prefix
}

/// The key of the entry the nucleus ends at: the probability mass (`probs`) of the entries
/// above it is under `target` and its own mass takes it there. The same select as
/// [`rank_key`] over the mass instead of the count; when no entry gets there, everything is
/// kept.
#[cube]
fn nucleus_key(
    vals: &Shared<[f32]>,
    probs: &Shared<[f32]>,
    mass: &mut Shared<[Atomic<f32>]>,
    found: &mut Shared<[u32]>,
    found_mass: &mut Shared<[f32]>,
    target: f32,
    #[comptime] vocab: usize,
) -> u32 {
    let unit = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    let mut prefix = 0u32.runtime();
    let mut mask = 0u32.runtime();
    let mut remaining = target;
    for round in 0..4u32 {
        let shift = 24u32 - 8u32 * round;
        for b in range_stepped(unit, 256usize, units) {
            mass[b].store(0.0f32);
        }
        if UNIT_POS == 0 {
            found[0] = 0u32;
            found_mass[0] = remaining;
        }
        sync_cube();
        for i in range_stepped(unit, vocab, units) {
            let p = probs[i];
            if p > 0.0f32 {
                let key = ordered_key(vals[i]);
                if key & mask == prefix {
                    mass[((key >> shift) & 0xFFu32) as usize].fetch_add(p);
                }
            }
        }
        sync_cube();
        if PLANE_POS == 0 {
            let per = 256u32 / PLANE_DIM;
            let top = 255u32 - UNIT_POS_PLANE * per;
            let mut local = 0.0f32.runtime();
            for j in 0..per {
                local += mass[(top - j) as usize].load();
            }
            let above = plane_exclusive_sum(local);
            if above < remaining && remaining <= above + local {
                let mut cum = above;
                let mut done = false.runtime();
                for j in 0..per {
                    if !done {
                        let bin = top - j;
                        let m = mass[bin as usize].load();
                        cum += m;
                        if cum >= remaining {
                            found[0] = bin;
                            found_mass[0] = remaining - (cum - m);
                            done = true;
                        }
                    }
                }
            }
        }
        sync_cube();
        prefix |= found[0] << shift;
        remaining = found_mass[0];
        mask |= 0xFFu32 << shift;
        sync_cube();
    }
    prefix
}

/// The maximum over the cube of `value`, one per unit; `scratch` holds one slot per plane.
#[cube]
fn cube_max(value: f32, scratch: &mut Shared<[f32]>) -> f32 {
    let m = plane_max(value);
    if UNIT_POS_PLANE == 0 {
        scratch[PLANE_POS as usize] = m;
    }
    sync_cube();
    let planes = CUBE_DIM / PLANE_DIM;
    let mut result = scratch[0];
    for p in 1..planes {
        result = max(result, scratch[p as usize]);
    }
    sync_cube();
    result
}

/// The sum over the cube of `value`, one per unit; `scratch` holds one slot per plane.
#[cube]
fn cube_sum(value: f32, scratch: &mut Shared<[f32]>) -> f32 {
    let s = plane_sum(value);
    if UNIT_POS_PLANE == 0 {
        scratch[PLANE_POS as usize] = s;
    }
    sync_cube();
    let planes = CUBE_DIM / PLANE_DIM;
    let mut result = 0.0f32.runtime();
    for p in 0..planes {
        result += scratch[p as usize];
    }
    sync_cube();
    result
}

/// One cube draws one token, see [`SamplingBackend::draw_token`]. `greedy` is 0 or 1.
#[cube(launch)]
fn draw_token_kernel(
    logits: &Tensor<f32>,
    noise: &Tensor<f32>,
    out: &mut Tensor<i32>,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    greedy: u32,
    #[comptime] vocab: usize,
) {
    let mut vals = Shared::<[f32]>::new_slice(vocab);
    let mut probs = Shared::<[f32]>::new_slice(vocab);
    let mut hist = Shared::<[Atomic<u32>]>::new_slice(256usize);
    let mut mass = Shared::<[Atomic<f32>]>::new_slice(256usize);
    let mut scratch = Shared::<[f32]>::new_slice(32usize);
    let mut scratch_index = Shared::<[u32]>::new_slice(32usize);
    let mut found = Shared::<[u32]>::new_slice(2usize);
    let mut found_mass = Shared::<[f32]>::new_slice(1usize);

    let unit = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    for i in range_stepped(unit, vocab, units) {
        vals[i] = logits[i] / temperature;
    }
    sync_cube();

    // The keys at or above `key_k` are the top-k, those at or above `key_p` the nucleus.
    let mut key_k = 0u32.runtime();
    let mut key_p = 0u32.runtime();
    if greedy == 0u32 {
        if top_k > 0u32 && top_k < vocab as u32 {
            key_k = rank_key(&vals, &mut hist, &mut found, top_k, vocab);
        }
        if top_p > 0.0f32 && top_p < 1.0f32 {
            // The softmax of the top-k entries, whose mass the nucleus is a share of.
            let mut local_max = (-3.0e38f32).runtime();
            for i in range_stepped(unit, vocab, units) {
                if ordered_key(vals[i]) >= key_k {
                    local_max = max(local_max, vals[i]);
                }
            }
            let max = cube_max(local_max, &mut scratch);
            let mut local_sum = 0.0f32.runtime();
            for i in range_stepped(unit, vocab, units) {
                let mut p = 0.0f32.runtime();
                if ordered_key(vals[i]) >= key_k {
                    p = (vals[i] - max).exp();
                }
                probs[i] = p;
                local_sum += p;
            }
            let sum = cube_sum(local_sum, &mut scratch);
            sync_cube();
            key_p = nucleus_key(
                &vals,
                &probs,
                &mut mass,
                &mut found,
                &mut found_mass,
                top_p * sum,
                vocab,
            );
        }
    }

    // The draw: the largest kept logit, plus Gumbel noise unless greedy. Ties go to the
    // lowest index, as an argmax would.
    let mut best = (-3.0e38f32).runtime();
    let mut best_index = 0u32.runtime();
    for i in range_stepped(unit, vocab, units) {
        let key = ordered_key(vals[i]);
        if key >= key_k && key >= key_p {
            let mut y = vals[i];
            if greedy == 0u32 {
                let u = min(max(noise[i], 1e-20f32), 0.999_999f32);
                y += -(-u.ln()).ln();
            }
            if y > best {
                best = y;
                best_index = i as u32;
            }
        }
    }
    let plane_best = plane_max(best);
    let candidate = select(best == plane_best, best_index, 0xFFFF_FFFFu32);
    let plane_index = plane_min(candidate);
    if UNIT_POS_PLANE == 0 {
        scratch[PLANE_POS as usize] = plane_best;
        scratch_index[PLANE_POS as usize] = plane_index;
    }
    sync_cube();
    if UNIT_POS == 0 {
        let planes = CUBE_DIM / PLANE_DIM;
        let mut best = scratch[0];
        let mut best_index = scratch_index[0];
        for p in 1..planes {
            let value = scratch[p as usize];
            let index = scratch_index[p as usize];
            if value > best || (value == best && index < best_index) {
                best = value;
                best_index = index;
            }
        }
        out[0] = i32::cast_from(best_index);
    }
}
