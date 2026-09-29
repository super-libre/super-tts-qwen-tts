//! Sampling of the codec tokens, on the device.
//!
//! A frame is sampled sixteen times, once per codebook, and each sample decides the input of
//! the next pass. Sampling on the host would make each of those a round trip that drains the
//! device: sampling where the logits are lets the passes of a frame queue up back to back, with
//! a single read of the sampled codes at the end of the frame.
//!
//! The vocabularies are small (2048 and 3072 entries) and the device has no sort, so the top-k
//! and top-p filters compare every entry with every other one instead: the rank of an entry is
//! the number of larger entries, the nucleus of an entry the probability mass of the larger
//! ones. A few million comparisons per sample, done by one reduction kernel. The draw itself is
//! the Gumbel-max trick: adding Gumbel noise to the logits and taking the largest one samples
//! from their softmax, without a prefix sum over the probabilities.

use burn::prelude::*;
use burn::tensor::activation::softmax;
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether [`Sampling::draw`] uses the kernel where it can.
static KERNEL: AtomicBool = AtomicBool::new(true);

/// Makes [`Sampling::draw`] use the tensor operations everywhere when `enabled` is false: the
/// slower path, kept for comparison.
pub fn use_kernel(enabled: bool) {
    KERNEL.store(enabled, Ordering::Relaxed);
}

/// What a filter puts in place of the logits it removes. Not `-inf`, which would leave `NaN`s
/// behind if it ever met a `+inf`, but as good as it for the softmax and the comparisons.
pub(crate) const MASKED: f32 = -1e30;

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    TopP { p: f32, temperature: f32 },
    TopKThenTopP { k: usize, p: f32, temperature: f32 },
}

impl Sampling {
    /// Draws one token out of `logits`, shape (vocab,), and returns its index as a one-element
    /// tensor. `noise` holds one uniform number in (0, 1) per entry, fresh for every draw.
    ///
    /// One kernel does it on the GPU backends, see [`crate::qwen3::sampling_kernel`]; elsewhere it is
    /// the tensor operations of [`Sampling::draw_ops`].
    pub fn draw(&self, logits: Tensor<1>, noise: Tensor<1>) -> Tensor<1, Int> {
        #[cfg(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "wgpu"
        ))]
        if KERNEL.load(Ordering::Relaxed)
            && crate::qwen3::sampling_kernel::available(&logits.device())
        {
            let (temperature, top_k, top_p, greedy) = match *self {
                Sampling::ArgMax => (1., 0, 1., true),
                Sampling::TopP { p, temperature } => (temperature, 0, p, false),
                Sampling::TopKThenTopP { k, p, temperature } => (temperature, k, p, false),
            };
            return crate::qwen3::sampling_kernel::draw_token(
                logits,
                noise,
                temperature,
                top_k,
                top_p,
                greedy,
            );
        }
        self.draw_ops(logits, noise)
    }

    /// The same sampling at another temperature.
    ///
    /// [`Sampling::ArgMax`] has none to set: it takes the largest logit, and
    /// dividing every logit by the same positive number does not change which
    /// one that is.
    #[must_use]
    pub fn with_temperature(self, temperature: f32) -> Self {
        match self {
            Sampling::ArgMax => Sampling::ArgMax,
            Sampling::TopP { p, .. } => Sampling::TopP { p, temperature },
            Sampling::TopKThenTopP { k, p, .. } => Sampling::TopKThenTopP { k, p, temperature },
        }
    }

    /// [`Sampling::draw`] as tensor operations.
    pub fn draw_ops(&self, logits: Tensor<1>, noise: Tensor<1>) -> Tensor<1, Int> {
        match *self {
            Sampling::ArgMax => logits.argmax(0),
            Sampling::TopP { p, temperature } => {
                let scaled = keep_top_p(logits.div_scalar(temperature), p);
                gumbel_argmax(scaled, noise)
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let scaled = keep_top_k(logits.div_scalar(temperature), k);
                let scaled = keep_top_p(scaled, p);
                gumbel_argmax(scaled, noise)
            }
        }
    }
}

/// Which entries are larger than which: `[i][j]` is 1 where `values[j] > values[i]` and 0
/// elsewhere. Computed as floats rather than as booleans converted afterwards: the conversion
/// is not an operation the fusion engine handles, and with it the comparison would be written
/// out as a matrix and read back, instead of being folded into the sum that follows.
fn larger(values: &Tensor<1>) -> Tensor<2> {
    let n = values.dims()[0];
    let difference = values.clone().reshape([1, n]) - values.clone().reshape([n, 1]);
    difference.mul_scalar(1e30).clamp(0., 1.)
}

/// Masks out everything but the `k` largest entries; ties at the boundary are all kept.
fn keep_top_k(values: Tensor<1>, k: usize) -> Tensor<1> {
    let n = values.dims()[0];
    if k == 0 || k >= n {
        return values;
    }
    let rank = larger(&values).sum_dim(1).reshape([n]);
    values.mask_fill(rank.greater_equal_elem(k as f32), MASKED)
}

/// Nucleus filtering: masks out the entries that the probability mass of the larger ones
/// already puts past `p`.
fn keep_top_p(values: Tensor<1>, p: f32) -> Tensor<1> {
    if p <= 0. || p >= 1. {
        return values;
    }
    let n = values.dims()[0];
    let probs = softmax(values.clone(), 0).reshape([1, n]);
    let above = (larger(&values) * probs).sum_dim(1).reshape([n]);
    values.mask_fill(above.greater_equal_elem(p), MASKED)
}

/// The index of the largest of `values + gumbel(noise)`, which is distributed as the softmax
/// of `values`.
fn gumbel_argmax(values: Tensor<1>, noise: Tensor<1>) -> Tensor<1, Int> {
    let gumbel = noise.clamp(1e-20, 0.999_999).log().neg().log().neg();
    (values + gumbel).argmax(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Distribution};

    fn device() -> Device {
        crate::qwen3::test_device()
    }

    fn logits(device: &Device) -> Tensor<1> {
        Tensor::from_data([0.0f32, 3.0, 1.0, 2.5, -4.0, 2.0, 0.5, -1.0], device)
    }

    fn histogram(sampling: &Sampling, draws: usize, device: &Device) -> Vec<usize> {
        device.seed(299_792_458);
        let logits = logits(device);
        let n = logits.dims()[0];
        let mut counts = vec![0; n];
        for _ in 0..draws {
            let noise = Tensor::random([n], Distribution::Uniform(0., 1.), (device, DType::F32));
            let code = sampling.draw(logits.clone(), noise).into_scalar::<i64>();
            counts[code as usize] += 1;
        }
        counts
    }

    #[test]
    fn argmax_takes_the_largest() {
        let device = device();
        let noise = Tensor::zeros([8], (&device, DType::F32));
        let code = Sampling::ArgMax
            .draw(logits(&device), noise)
            .into_scalar::<i64>();
        assert_eq!(code, 1);
    }

    #[test]
    fn top_k_keeps_the_k_largest() {
        let device = device();
        let sampling = Sampling::TopKThenTopP {
            k: 3,
            p: 1.0,
            temperature: 1.0,
        };
        let counts = histogram(&sampling, 400, &device);
        // 3.0, 2.5 and 2.0 are the three largest.
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count > 0, matches!(i, 1 | 3 | 5), "entry {i}: {count}");
        }
        assert!(counts[1] > counts[3] && counts[3] > counts[5], "{counts:?}");
    }

    #[test]
    fn top_p_keeps_the_nucleus() {
        let device = device();
        // softmax(logits): 3.0 -> 0.44, 2.5 -> 0.27, 2.0 -> 0.16, ...: the mass above 2.5 is
        // 0.44 < 0.6 and the mass above 2.0 is 0.71 >= 0.6, so the first two survive.
        let sampling = Sampling::TopP {
            p: 0.6,
            temperature: 1.0,
        };
        let counts = histogram(&sampling, 400, &device);
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count > 0, matches!(i, 1 | 3), "entry {i}: {count}");
        }
    }

    /// The kernel and the operations draw the same token from the same noise, but where a
    /// threshold falls between two entries that round differently.
    #[cfg(any(
        feature = "cuda",
        feature = "rocm",
        feature = "metal",
        feature = "vulkan",
        feature = "wgpu"
    ))]
    #[test]
    fn kernel_matches_operations() {
        let device = device();
        if !crate::qwen3::sampling_kernel::available(&device) {
            return;
        }
        device.seed(7);
        let cases = [
            (Sampling::ArgMax, 0),
            (
                Sampling::TopP {
                    p: 0.9,
                    temperature: 0.8,
                },
                2,
            ),
            (
                Sampling::TopKThenTopP {
                    k: 50,
                    p: 1.0,
                    temperature: 0.9,
                },
                2,
            ),
            (
                Sampling::TopKThenTopP {
                    k: 50,
                    p: 0.9,
                    temperature: 0.9,
                },
                2,
            ),
        ];
        for (vocab, (sampling, allowed)) in [2048, 3072].into_iter().zip(cases.iter().cycle()) {
            let mut mismatches = 0;
            for _ in 0..100 {
                let logits =
                    Tensor::random([vocab], Distribution::Normal(0., 3.), (&device, DType::F32));
                let noise = Tensor::random(
                    [vocab],
                    Distribution::Uniform(0., 1.),
                    (&device, DType::F32),
                );
                let kernel = sampling
                    .draw(logits.clone(), noise.clone())
                    .into_scalar::<i64>();
                let ops = sampling.draw_ops(logits, noise).into_scalar::<i64>();
                if kernel != ops {
                    mismatches += 1;
                }
            }
            assert!(
                mismatches <= *allowed,
                "{sampling:?} over {vocab} entries: {mismatches} of 100 draws differ"
            );
        }
    }

    #[test]
    fn gumbel_samples_the_softmax() {
        let device = device();
        let sampling = Sampling::TopP {
            p: 1.0,
            temperature: 2.0,
        };
        let draws = 4000;
        let counts = histogram(&sampling, draws, &device);
        let probs = softmax(logits(&device).div_scalar(2.0), 0)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (i, (&count, &p)) in counts.iter().zip(&probs).enumerate() {
            let expected = p * draws as f32;
            let tolerance = 5. * (expected * (1. - p)).sqrt() + 3.;
            assert!(
                (count as f32 - expected).abs() < tolerance,
                "entry {i}: {count} draws, {expected} expected"
            );
        }
    }
}
