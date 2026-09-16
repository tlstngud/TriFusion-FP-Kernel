import torch
from torch import nn
import torch.nn.functional as F
from .kernels import RMSNorm
from .masking import time_mask


class CNNStage(nn.Module):
    def __init__(self, cin, cout, stride, backend):
        super().__init__()
        self.stride = stride
        self.backend = backend
        self.buffer_fusion = False
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=(stride, 2), padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.skip = nn.Conv2d(cin, cout, 1, stride=(stride, 2), bias=False)
        self.norm1, self.norm2 = RMSNorm(cout, backend=backend), RMSNorm(cout, backend=backend)

    def forward(self, x, lengths):
        lengths = (lengths + self.stride - 1) // self.stride
        h = self.conv1(x)
        m = time_mask(lengths, h.shape[2])[:, None, :, None]

        def clean(v):
            return v.masked_fill(~m, 0)

        def norm(v, layer):
            return layer(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.backend == "l4" and self.buffer_fusion:
            from .kernels.buffers import front_norm

            h = front_norm(h, lengths, self.norm1, silu=True)
            h = front_norm(self.conv2(h), lengths, self.norm2)
        elif self.backend == "l4":
            # RMSNorm has no bias and SiLU(0)=0. A zeroed convolution output
            # stays zero through both; avoid two full activation copies.
            h = F.silu(norm(clean(h), self.norm1))
            h = norm(clean(self.conv2(h)), self.norm2)
        else:
            h = clean(F.silu(clean(norm(clean(h), self.norm1))))
            h = clean(norm(clean(self.conv2(h)), self.norm2))
        return clean(F.silu(h + clean(self.skip(x))))


class Frontend(nn.Module):
    def __init__(self, backend="l4", channels_last=True, checkpoint_stages=False):
        super().__init__()
        self.backend, self.native_kernels = backend, False
        self.register_buffer("window", torch.hann_window(400, periodic=True), persistent=False)
        self.stages = nn.ModuleList(
            [
                CNNStage(1, 32, 1, backend),
                CNNStage(32, 64, 2, backend),
                CNNStage(64, 128, 2, backend),
            ]
        )
        self.channels_last, self.checkpoint_stages = channels_last, checkpoint_stages
        if channels_last:
            self.to(memory_format=torch.channels_last)

    def main_lengths(self, sample_lengths):
        """Compute the valid output grid on the metadata's original device."""
        lengths = sample_lengths // 160 + 1
        for stage in self.stages:
            lengths = (lengths + stage.stride - 1) // stage.stride
        return lengths

    def spectrogram(self, audio, sample_lengths, channels):
        b, c, n = audio.shape
        with torch.autocast(device_type=audio.device.type, enabled=False):
            m = time_mask(sample_lengths, n)[:, None, :]
            cm = torch.arange(c, device=audio.device)[None, :] < channels[:, None]
            wave = audio.float().masked_fill(~m, 0).masked_fill(~cm[..., None], 0)
            spec = torch.stft(
                wave.reshape(b * c, n),
                n_fft=512,
                hop_length=160,
                win_length=400,
                window=self.window.float(),
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            # |z|^2 = real^2 + imag^2: avoid a square root followed by a square.
            power = (spec.real.square() + spec.imag.square()).reshape(b, c, 257, -1).sum(
                1
            ) / channels[:, None, None]
            log = torch.log(1e-6 + power).transpose(1, 2).unsqueeze(1)
            lengths = sample_lengths // 160 + 1
            log = log.masked_fill(~time_mask(lengths, log.shape[2])[:, None, :, None], 0)
        return log, lengths

    def bands(self, x):
        if self.native_kernels and self.backend == "l4":
            from .kernels.native import bands

            return bands(x)
        # Pool only frequency; preserve each individual time step.
        x = F.adaptive_avg_pool2d(x, (x.shape[2], 8))
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, audio, sample_lengths, channels):
        x, lengths = self.spectrogram(audio, sample_lengths, channels)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        fine = fine_lengths = None
        for i, stage in enumerate(self.stages):
            x = stage(x, lengths)
            lengths = (lengths + stage.stride - 1) // stage.stride
            if i == 1:
                fine, fine_lengths = self.bands(x), lengths
        main = self.bands(x).flatten(2)
        return main, fine, lengths, fine_lengths
