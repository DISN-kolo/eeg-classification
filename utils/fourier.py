import numpy as np
from scipy import signal
from scipy.signal import ShortTimeFFT


def compute_spectrogram_stft(
        data_uv,
        sfreq,
        window_seconds=1.0,
        overlap_ratio=0.5,
        fmax=60.0):
    n_per_seg = int(window_seconds * sfreq)
    n_overlap = int(n_per_seg * overlap_ratio)

    sft = ShortTimeFFT.from_window(
        ("hann"),
        sfreq,
        n_per_seg,
        n_overlap,
        fft_mode="onesided2X",
        scale_to="psd",
    )

    n_samples = data_uv.shape[-1]
    n_segments = 1 + (n_samples - n_per_seg) // sft.hop
    k_offset = sft.m_num_mid

    freqs = sft.f
    times = sft.t(n_samples, p0=0, p1=n_segments, k_offset=k_offset)
    sxx = sft.spectrogram(
        data_uv,
        detr="constant",
        p0=0,
        p1=n_segments,
        k_offset=k_offset,
    )

    if (fmax is not None):
        keep = freqs <= fmax
        freqs = freqs[keep]
        sxx = sxx[keep, :]

    power_db = 10 * np.log10(sxx + np.finfo(float).eps)

    return freqs, times, power_db
