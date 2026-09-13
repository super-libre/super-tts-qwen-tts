//! The ECAPA-TDNN speaker encoder of the Qwen3-TTS Base checkpoints.
//!
//! It turns a 24 kHz reference recording into the one embedding vector the talker is conditioned
//! on, see [`Voice::Embedding`](crate::qwen3::model::Voice::Embedding), out of the 128-bin log-mel
//! spectrogram of [`mel_spectrogram`]. The spectrogram is computed on the host: it runs once per
//! voice, and a few hundred FFTs of 1024 points are not worth a device kernel.
//!
//! See "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based
//! Speaker Verification" (<https://huggingface.co/papers/2005.07143>).

use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::ops::PadMode;

use crate::qwen3::config::SpeakerEncoderConfig;

/// A convolution with "same" reflect padding, followed by a ReLU.
#[derive(Module, Debug)]
struct TdnnBlock {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl TdnnBlock {
    fn init(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> Self {
        let total = dilation * (kernel_size - 1);
        let pad_left = total / 2;
        Self {
            conv: Conv1dConfig::new(in_c, out_c, kernel_size)
                .with_dilation(dilation)
                .init(device),
            pad_left,
            pad_right: total - pad_left,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let xs = if self.pad_left + self.pad_right > 0 {
            xs.pad([(self.pad_left, self.pad_right)], PadMode::Reflect)
        } else {
            xs
        };
        relu(self.conv.forward(xs))
    }
}

/// Res2Net: the channels are split in `scale` groups, and every group but the first goes
/// through its own convolution, summed with the output of the previous group.
#[derive(Module, Debug)]
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn init(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> Self {
        let in_channel = in_channels / scale;
        let hidden_channel = out_channels / scale;
        Self {
            blocks: (0..scale - 1)
                .map(|_| TdnnBlock::init(in_channel, hidden_channel, kernel_size, dilation, device))
                .collect(),
            scale,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut outputs: Vec<Tensor<3>> = Vec::with_capacity(self.scale);
        for (i, chunk) in xs.chunk(self.scale, 1).into_iter().enumerate() {
            let output = match i {
                0 => chunk,
                1 => self.blocks[0].forward(chunk),
                _ => self.blocks[i - 1].forward(chunk + outputs[i - 1].clone()),
            };
            outputs.push(output);
        }
        Tensor::cat(outputs, 1)
    }
}

/// Squeeze-and-excitation: rescales every channel by a factor computed from its mean over time.
#[derive(Module, Debug)]
struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    fn init(in_channels: usize, se_channels: usize, out_channels: usize, device: &Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(in_channels, se_channels, 1).init(device),
            conv2: Conv1dConfig::new(se_channels, out_channels, 1).init(device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let scale = self.conv1.forward(xs.clone().mean_dim(2));
        let scale = self.conv2.forward(relu(scale));
        xs * sigmoid(scale)
    }
}

#[derive(Module, Debug)]
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TdnnBlock,
    se_block: SqueezeExcitationBlock,
}

impl SeRes2NetBlock {
    fn init(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> Self {
        Self {
            tdnn1: TdnnBlock::init(in_channels, out_channels, 1, 1, device),
            res2net_block: Res2NetBlock::init(
                out_channels,
                out_channels,
                res2net_scale,
                kernel_size,
                dilation,
                device,
            ),
            tdnn2: TdnnBlock::init(out_channels, out_channels, 1, 1, device),
            se_block: SqueezeExcitationBlock::init(out_channels, se_channels, out_channels, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let hidden = self.tdnn1.forward(xs.clone());
        let hidden = self.res2net_block.forward(hidden);
        let hidden = self.tdnn2.forward(hidden);
        xs + self.se_block.forward(hidden)
    }
}

/// Attentive statistics pooling: the attention-weighted mean and standard deviation of every
/// channel over time, concatenated.
#[derive(Module, Debug)]
struct AttentiveStatisticsPooling {
    tdnn: TdnnBlock,
    conv: Conv1d,
}

impl AttentiveStatisticsPooling {
    const EPS: f32 = 1e-12;

    fn init(channels: usize, attention_channels: usize, device: &Device) -> Self {
        Self {
            tdnn: TdnnBlock::init(channels * 3, attention_channels, 1, 1, device),
            conv: Conv1dConfig::new(attention_channels, channels, 1).init(device),
        }
    }

    /// Weighted mean and standard deviation over time, (B, C, 1) each; `weights` sums to one
    /// over time.
    fn statistics(xs: &Tensor<3>, weights: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let mean = (xs.clone() * weights.clone()).sum_dim(2);
        let var = ((xs.clone() - mean.clone()).square() * weights).sum_dim(2);
        (mean, var.clamp_min(Self::EPS).sqrt())
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let [b, c, t] = xs.dims();
        let uniform = Tensor::<3>::ones([b, 1, t], &xs.device())
            .div_scalar(t as f32)
            .cast(xs.dtype());
        let (mean, std) = Self::statistics(&xs, uniform);
        let attention = Tensor::cat(
            vec![xs.clone(), mean.expand([b, c, t]), std.expand([b, c, t])],
            1,
        );
        let attention = self.conv.forward(tanh(self.tdnn.forward(attention)));
        let (mean, std) = Self::statistics(&xs, softmax(attention, 2));
        Tensor::cat(vec![mean, std], 1)
    }
}

/// The speaker encoder, the `speaker_encoder.` prefix of the `model.safetensors` file of a Base
/// checkpoint.
#[derive(Module, Debug)]
pub struct SpeakerEncoder {
    first: TdnnBlock,
    blocks: Vec<SeRes2NetBlock>,
    /// Multi-layer feature aggregation, over the concatenated outputs of the blocks.
    mfa: TdnnBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
}

impl SpeakerEncoder {
    pub(crate) fn init(cfg: &SpeakerEncoderConfig, device: &Device) -> Result<Self, String> {
        let n = cfg.enc_channels.len();
        if n != cfg.enc_kernel_sizes.len() || n != cfg.enc_dilations.len() {
            return Err(
                "enc_channels, enc_kernel_sizes and enc_dilations should have the same length"
                    .to_string(),
            );
        }
        // One TDNN block, at least one SE-Res2Net block and the feature aggregation.
        if n < 3 {
            return Err(format!(
                "enc_channels should have at least three entries, got {n}"
            ));
        }
        Ok(Self {
            first: TdnnBlock::init(
                cfg.mel_dim,
                cfg.enc_channels[0],
                cfg.enc_kernel_sizes[0],
                cfg.enc_dilations[0],
                device,
            ),
            blocks: (1..n - 1)
                .map(|i| {
                    SeRes2NetBlock::init(
                        cfg.enc_channels[i - 1],
                        cfg.enc_channels[i],
                        cfg.enc_res2net_scale,
                        cfg.enc_se_channels,
                        cfg.enc_kernel_sizes[i],
                        cfg.enc_dilations[i],
                        device,
                    )
                })
                .collect(),
            mfa: TdnnBlock::init(
                cfg.enc_channels[n - 1],
                cfg.enc_channels[n - 1],
                cfg.enc_kernel_sizes[n - 1],
                cfg.enc_dilations[n - 1],
                device,
            ),
            asp: AttentiveStatisticsPooling::init(
                cfg.enc_channels[n - 1],
                cfg.enc_attention_channels,
                device,
            ),
            fc: Conv1dConfig::new(cfg.enc_channels[n - 1] * 2, cfg.enc_dim, 1).init(device),
        })
    }

    /// The embeddings (B, enc_dim) of the log-mel spectrograms `mels` (B, frames, mel_dim).
    pub fn forward(&self, mels: Tensor<3>) -> Tensor<2> {
        let mut xs = self.first.forward(mels.swap_dims(1, 2));
        let mut outputs = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.iter() {
            xs = block.forward(xs);
            outputs.push(xs.clone());
        }
        let xs = self.mfa.forward(Tensor::cat(outputs, 1));
        self.fc.forward(self.asp.forward(xs)).squeeze_dim(2)
    }
}

/// Parameters of the log-mel spectrogram the speaker encoder expects.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub num_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1024,
            hop_size: 256,
            win_size: 1024,
            num_mels: 128,
            fmin: 0.,
            fmax: 12000.,
        }
    }
}

// The Slaney mel scale, as in librosa with `htk=False`.
fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200. / 3.;
    let min_log_hz = 1000.;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200. / 3.;
    let min_log_hz = 1000.;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// The mel filterbank, `num_mels` rows of `n_fft / 2 + 1`, matching `librosa.filters.mel` with
/// its default Slaney normalization.
pub fn mel_filters(cfg: &MelConfig) -> Vec<f32> {
    let n_freqs = cfg.n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * cfg.sample_rate as f64 / cfg.n_fft as f64)
        .collect();
    let (min_mel, max_mel) = (hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax));
    let n_points = cfg.num_mels + 2;
    let mel_f: Vec<f64> = (0..n_points)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (n_points - 1) as f64))
        .collect();
    let mut weights = vec![0f32; cfg.num_mels * n_freqs];
    for i in 0..cfg.num_mels {
        let enorm = 2. / (mel_f[i + 2] - mel_f[i]);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            weights[i * n_freqs + k] = (lower.min(upper).max(0.) * enorm) as f32;
        }
    }
    weights
}

/// In-place iterative radix-2 FFT; `re` and `im` must have a power of two length.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2. * std::f64::consts::PI / len as f64;
        let (w_re, w_im) = (angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let (mut cur_re, mut cur_im) = (1f64, 0f64);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (t_re, t_im) = (
                    re[b] * cur_re - im[b] * cur_im,
                    re[b] * cur_im + im[b] * cur_re,
                );
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
        }
        len <<= 1;
    }
}

/// The log-mel spectrogram of a mono recording sampled at `cfg.sample_rate`: `frames` rows of
/// `num_mels`, row-major, returned with `frames`.
///
/// This mirrors the reference `mel_spectrogram` function: reflect padding of `(n_fft - hop) / 2`
/// samples, a periodic Hann window, the magnitude spectrum, the Slaney mel filterbank and a
/// natural logarithm with a `1e-5` floor.
pub fn mel_spectrogram(samples: &[f32], cfg: &MelConfig) -> Result<(Vec<f32>, usize), String> {
    if !cfg.n_fft.is_power_of_two() {
        return Err(format!("n_fft must be a power of two, got {}", cfg.n_fft));
    }
    if cfg.hop_size == 0 || cfg.hop_size > cfg.n_fft {
        return Err(format!(
            "hop_size {} should be in 1..={}",
            cfg.hop_size, cfg.n_fft
        ));
    }
    if cfg.win_size > cfg.n_fft {
        return Err(format!(
            "win_size {} should be at most n_fft {}",
            cfg.win_size, cfg.n_fft
        ));
    }
    let padding = (cfg.n_fft - cfg.hop_size) / 2;
    let len = samples.len();
    // The padding reflects around the first and last samples, and one full window has to fit
    // in the padded signal.
    if len <= padding || len + 2 * padding < cfg.n_fft {
        return Err(format!("the recording is too short ({len} samples)"));
    }
    let mut padded = Vec::with_capacity(len + 2 * padding);
    padded.extend((1..=padding).rev().map(|i| samples[i] as f64));
    padded.extend(samples.iter().map(|&v| v as f64));
    padded.extend((1..=padding).map(|i| samples[len - 1 - i] as f64));
    let n_frames = (padded.len() - cfg.n_fft) / cfg.hop_size + 1;
    let n_freqs = cfg.n_fft / 2 + 1;
    let window: Vec<f64> = (0..cfg.win_size)
        .map(|i| 0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / cfg.win_size as f64).cos())
        .collect();
    let filters = mel_filters(cfg);
    let mut mels = vec![0f32; n_frames * cfg.num_mels];
    let mut re = vec![0f64; cfg.n_fft];
    let mut im = vec![0f64; cfg.n_fft];
    let mut magnitudes = vec![0f64; n_freqs];
    for frame in 0..n_frames {
        let start = frame * cfg.hop_size;
        re.iter_mut().for_each(|v| *v = 0.);
        im.iter_mut().for_each(|v| *v = 0.);
        for i in 0..cfg.win_size {
            re[i] = padded[start + i] * window[i];
        }
        fft(&mut re, &mut im);
        for k in 0..n_freqs {
            magnitudes[k] = (re[k] * re[k] + im[k] * im[k] + 1e-9).sqrt();
        }
        for m in 0..cfg.num_mels {
            let filter = &filters[m * n_freqs..(m + 1) * n_freqs];
            let v: f64 = filter
                .iter()
                .zip(magnitudes.iter())
                .map(|(&w, &s)| w as f64 * s)
                .sum();
            mels[frame * cfg.num_mels + m] = v.max(1e-5).ln() as f32;
        }
    }
    Ok((mels, n_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SpeakerEncoderConfig {
        SpeakerEncoderConfig {
            enc_dim: 32,
            mel_dim: 16,
            // The feature aggregation takes the concatenation of the SE-Res2Net outputs, so
            // the last entry is (n - 2) times the block width.
            enc_channels: vec![24, 24, 24, 48],
            enc_kernel_sizes: vec![3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 4,
            enc_se_channels: 8,
            sample_rate: 24000,
        }
    }

    #[test]
    fn forward_has_the_embedding_shape() {
        let device = crate::qwen3::test_device();
        let cfg = test_config();
        let model = SpeakerEncoder::init(&cfg, &device).unwrap();
        for batch_size in [1, 3] {
            let mels = Tensor::<3>::zeros([batch_size, 20, cfg.mel_dim], &device);
            assert_eq!(model.forward(mels).dims(), [batch_size, cfg.enc_dim]);
        }
    }

    #[test]
    fn degenerate_configs_are_rejected() {
        let device = crate::qwen3::test_device();
        let cfg = SpeakerEncoderConfig {
            enc_channels: vec![24],
            enc_kernel_sizes: vec![3],
            enc_dilations: vec![1],
            ..test_config()
        };
        assert!(SpeakerEncoder::init(&cfg, &device).is_err());
        let samples = vec![0f32; 4096];
        let cfg = MelConfig {
            hop_size: 0,
            ..Default::default()
        };
        assert!(mel_spectrogram(&samples, &cfg).is_err());
        let cfg = MelConfig {
            win_size: 4096,
            ..Default::default()
        };
        assert!(mel_spectrogram(&samples, &cfg).is_err());
        let cfg = MelConfig::default();
        assert!(mel_spectrogram(&samples[..16], &cfg).is_err());
    }

    /// A pure tone lands its energy in the mel bins around its frequency.
    #[test]
    fn tone_lands_in_its_band() {
        let cfg = MelConfig::default();
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2. * std::f32::consts::PI * 1000. * i as f32 / 24000.).sin())
            .collect();
        let (mels, frames) = mel_spectrogram(&samples, &cfg).unwrap();
        let padded = samples.len() + 2 * ((cfg.n_fft - cfg.hop_size) / 2);
        assert_eq!(frames, (padded - cfg.n_fft) / cfg.hop_size + 1);
        let frame = &mels[10 * cfg.num_mels..11 * cfg.num_mels];
        let (loudest, _) = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        // The filters are centered on the mel points 1..=num_mels of an even split of the
        // scale: 1 kHz is mel 15 of the 51 that reach 12 kHz, so around bin 37.
        let (min_mel, max_mel) = (hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax));
        let step = (max_mel - min_mel) / (cfg.num_mels + 1) as f64;
        let expected = ((hz_to_mel(1000.) - min_mel) / step).round() as usize - 1;
        assert!(
            loudest.abs_diff(expected) <= 1,
            "loudest bin {loudest}, expected {expected}"
        );
    }
}
