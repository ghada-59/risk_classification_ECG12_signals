"""QRS / HRV / morphology features extracted from lead II.

Wrapper around ``neurokit2.ecg_process`` with a robust fallback: if the
auto-pipeline fails (very noisy snapshot, no detectable peaks), we return
a zero vector and a `valid=False` flag so the downstream model still
sees a well-shaped tensor.

The output is a fixed-length ``float32`` vector of 12 features:
    [hr_mean, hr_std, sdnn, rmssd, pnn50,
     qrs_width_mean, qrs_width_std,
     pr_interval_mean, qt_interval_mean,
     st_level_mean, t_amplitude_mean,
     n_beats]
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from .config import SAMPLING_RATE_HZ


logger = logging.getLogger(__name__)

N_FEATURES = 12
LEAD_II_INDEX = 1


@dataclass(frozen=True)
class ECGFeatures:
    vector: np.ndarray
    valid: bool

    def as_array(self) -> np.ndarray:
        return self.vector.astype(np.float32, copy=False)


def _zero_vector(valid: bool = False) -> ECGFeatures:
    return ECGFeatures(np.zeros(N_FEATURES, dtype=np.float32), valid)


def _safe_mean(arr) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else 0.0


def _safe_std(arr) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std()) if a.size > 1 else 0.0


def extract_features(
    signal_12xN: np.ndarray, sampling_rate: int = SAMPLING_RATE_HZ
) -> ECGFeatures:
    """Compute the 12-dim feature vector for a single ECG window.

    ``signal_12xN`` must be ``(12, n_samples)`` in mV or z-scored units.
    Defensive against any neurokit2 failure mode — always returns a
    well-shaped float32 vector (zeros if peaks could not be detected).
    """
    try:
        return _extract_features_impl(signal_12xN, sampling_rate)
    except Exception as exc:
        logger.debug("extract_features failed: %s", exc)
        return _zero_vector()


def _extract_features_impl(
    signal_12xN: np.ndarray, sampling_rate: int
) -> ECGFeatures:
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise RuntimeError("neurokit2 not installed") from exc

    if signal_12xN.ndim != 2 or signal_12xN.shape[0] != 12:
        return _zero_vector()

    lead_ii = signal_12xN[LEAD_II_INDEX].astype(np.float64)
    if not np.isfinite(lead_ii).all():
        return _zero_vector()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            signals, info = nk.ecg_process(lead_ii, sampling_rate=sampling_rate)
        except Exception as exc:
            logger.debug("ecg_process failed: %s", exc)
            return _zero_vector()

    r_raw = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)
    r_peaks = r_raw[np.isfinite(r_raw)].astype(int)
    if r_peaks.size < 2:
        return _zero_vector()

    rr = np.diff(r_peaks) / sampling_rate
    hr_inst = 60.0 / rr
    hr_mean = _safe_mean(hr_inst)
    hr_std = _safe_std(hr_inst)
    sdnn = _safe_std(rr * 1000.0)
    rr_ms = rr * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if rr_ms.size > 1 else 0.0
    nn50 = int(np.sum(np.abs(np.diff(rr_ms)) > 50)) if rr_ms.size > 1 else 0
    pnn50 = 100.0 * nn50 / max(1, len(rr_ms) - 1)

    q_peaks = np.asarray(info.get("ECG_Q_Peaks", []), dtype=float)
    s_peaks = np.asarray(info.get("ECG_S_Peaks", []), dtype=float)
    p_peaks = np.asarray(info.get("ECG_P_Peaks", []), dtype=float)
    t_peaks = np.asarray(info.get("ECG_T_Peaks", []), dtype=float)

    qrs_widths: list[float] = []
    n = min(len(q_peaks), len(s_peaks))
    for i in range(n):
        q, s = q_peaks[i], s_peaks[i]
        if np.isfinite(q) and np.isfinite(s) and s > q:
            qrs_widths.append((s - q) / sampling_rate * 1000.0)
    qrs_width_mean = _safe_mean(qrs_widths)
    qrs_width_std = _safe_std(qrs_widths)

    pr_intervals: list[float] = []
    n = min(len(p_peaks), len(r_peaks))
    for i in range(n):
        p, r = p_peaks[i], r_peaks[i]
        if np.isfinite(p) and np.isfinite(r) and r > p:
            pr_intervals.append((r - p) / sampling_rate * 1000.0)
    pr_mean = _safe_mean(pr_intervals)

    qt_intervals: list[float] = []
    n = min(len(q_peaks), len(t_peaks))
    for i in range(n):
        q, t = q_peaks[i], t_peaks[i]
        if np.isfinite(q) and np.isfinite(t) and t > q:
            qt_intervals.append((t - q) / sampling_rate * 1000.0)
    qt_mean = _safe_mean(qt_intervals)

    cleaned = np.asarray(signals.get("ECG_Clean", lead_ii))
    st_offset = int(0.08 * sampling_rate)
    st_levels: list[float] = []
    for s in s_peaks:
        idx = int(s) + st_offset
        if np.isfinite(s) and 0 <= idx < len(cleaned):
            st_levels.append(float(cleaned[idx]))
    st_level_mean = _safe_mean(st_levels)

    t_amps = [float(cleaned[int(t)]) for t in t_peaks
              if np.isfinite(t) and 0 <= int(t) < len(cleaned)]
    t_amp_mean = _safe_mean(t_amps)

    vec = np.array(
        [
            hr_mean,
            hr_std,
            sdnn,
            rmssd,
            pnn50,
            qrs_width_mean,
            qrs_width_std,
            pr_mean,
            qt_mean,
            st_level_mean,
            t_amp_mean,
            float(r_peaks.size),
        ],
        dtype=np.float32,
    )
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return ECGFeatures(vec, valid=True)


def _self_test() -> None:
    import neurokit2 as nk

    sig = nk.ecg_simulate(duration=10, sampling_rate=SAMPLING_RATE_HZ, heart_rate=72)
    twelve = np.tile(sig, (12, 1)).astype(np.float32)
    feats = extract_features(twelve)
    print(f"valid={feats.valid}  shape={feats.vector.shape}")
    print(f"vector={feats.vector}")
    assert feats.valid
    assert feats.vector.shape == (N_FEATURES,)
    assert 50 < feats.vector[0] < 120, f"HR out of expected range: {feats.vector[0]}"
    print("feature_extraction self-test OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
