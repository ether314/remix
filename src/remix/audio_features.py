"""Measured facts about a soundtrack. Claude cannot listen, so these are what it gets instead."""
from __future__ import annotations

import numpy as np

ANALYSIS_RATE = 22050
TEMPO_SECONDS = 240  # enough to find a beat, and it bounds the work on long videos
TEMPO_RANGE = (60.0, 200.0)
HOP = 256


def active_ratio(samples: np.ndarray, rate: int, floor_dbfs: float = -45.0, window: float = 0.5) -> float:
    """Share of half-second windows whose RMS level is above the floor."""
    size = int(rate * window)
    count = len(samples) // size
    if count == 0:
        return 0.0
    frames = samples[: count * size].reshape(count, size).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.mean(20 * np.log10(np.maximum(rms, 1e-9)) > floor_dbfs))


def onset_strength(samples: np.ndarray, size: int = 2048, hop: int = HOP) -> np.ndarray:
    """Spectral flux: how much energy appears between one frame and the next."""
    if len(samples) < size + hop:
        return np.zeros(0, dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, size)[::hop]
    window = np.hanning(size).astype(np.float32)
    flux = np.empty(len(frames) - 1, dtype=np.float32)
    previous = None
    for start in range(0, len(frames), 1024):  # in blocks, so memory stays small
        block = np.log1p(100 * np.abs(np.fft.rfft(frames[start : start + 1024] * window, axis=1)))
        joined = block if previous is None else np.vstack([previous, block])
        rise = np.maximum(np.diff(joined, axis=0), 0).sum(axis=1)
        offset = start if previous is None else start - 1
        flux[offset : offset + len(rise)] = rise
        previous = block[-1:]
    return flux


def estimate_tempo(samples: np.ndarray, rate: int = ANALYSIS_RATE) -> dict | None:
    """Tempo from the autocorrelation of onset strength, or None when there is no steady beat.

    Half-time and double-time answers are common with this method: treat it as an estimate.
    """
    envelope = onset_strength(samples[: rate * TEMPO_SECONDS])
    fps = rate / HOP
    if len(envelope) < fps * 8:
        return None

    kernel = max(3, int(fps * 0.5) | 1)
    trend = np.convolve(envelope, np.ones(kernel) / kernel, mode="same")
    envelope = np.maximum(envelope - trend, 0)  # keep the peaks, drop the slow loudness trend
    if not envelope.any():
        return None
    envelope = envelope - envelope.mean()
    autocorr = np.fft.irfft(np.abs(np.fft.rfft(envelope, n=2 * len(envelope))) ** 2)[: len(envelope)]
    if autocorr[0] <= 0:
        return None
    autocorr = autocorr / autocorr[0]

    shortest = int(np.floor(fps * 60 / TEMPO_RANGE[1]))
    longest = min(int(np.ceil(fps * 60 / TEMPO_RANGE[0])), len(autocorr) - 2)
    lags = np.arange(shortest, longest + 1)
    prior = np.exp(-0.5 * np.log2((60 * fps / lags) / 120.0) ** 2)  # mild pull toward 120 BPM
    best = int(lags[np.argmax(autocorr[lags] * prior)])
    strength = float(autocorr[best])
    if strength < 0.1:
        return None

    before, peak, after = autocorr[best - 1], autocorr[best], autocorr[best + 1]
    curve = before - 2 * peak + after
    shift = 0.5 * (before - after) / curve if curve else 0.0  # sub-frame peak position
    return {"bpm": round(float(60 * fps / (best + shift)), 1), "strength": round(strength, 2)}
