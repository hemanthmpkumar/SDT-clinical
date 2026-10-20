"""
Modality Encoders for SCT
==========================
Lightweight NumPy/SciPy implementations of all five vertex encoders.
These are principled featurizers (no deep learning required to run).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import dct


D_EMB = 64   # Embedding dimension (reduced from 256 for CPU efficiency)


# -----------------------------------------------------------------------
# Spectral Encoder  (SpecFormer stand-in)
# -----------------------------------------------------------------------

class SpectralEncoder:
    """
    Encodes a transmission spectrum into a D_EMB-dimensional vector.

    Features:
      - Wavelength-binned mean absorption depths (10 macro-bins)
      - Band ratios capturing key molecular features
      - Rayleigh slope estimate
      - DCT coefficients of normalized spectrum (for smooth representation)
    """

    FEATURE_BINS = [
        (0.5,  1.0),   # optical (Na, K)
        (1.0,  1.5),   # H₂O J-band
        (1.5,  2.0),   # H₂O H-band
        (2.0,  2.5),   # CO₂ + CO
        (2.5,  3.5),   # H₂O K-band + CO₂
        (3.5,  4.5),   # CO₂ 4.3μm + SO₂ 4.05μm
        (4.5,  6.0),   # H₂O far-IR + SO₂
        (6.0,  8.5),   # SO₂ 7.7μm + silicates
        (8.5, 12.0),   # silicates
        (0.5, 12.0),   # broad mean
    ]

    def __init__(self, d: int = D_EMB):
        self.d = d

    def encode(
        self,
        wavelengths: np.ndarray,
        depths: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        feats = []

        # 1. Bin-averaged depths (10 bins)
        for wl_lo, wl_hi in self.FEATURE_BINS:
            mask = (wavelengths >= wl_lo) & (wavelengths < wl_hi)
            if mask.sum() > 0:
                w = 1.0 / (uncertainties[mask] ** 2 + 1e-20)
                feats.append(float(np.average(depths[mask], weights=w)))
            else:
                feats.append(0.0)

        # 2. SO₂ / H₂O amplitude ratio
        so2_mask = (wavelengths >= 4.0) & (wavelengths < 4.2)
        h2o_mask = (wavelengths >= 1.3) & (wavelengths < 1.5)
        so2_amp = float(depths[so2_mask].mean()) if so2_mask.sum() > 0 else 0.0
        h2o_amp = float(depths[h2o_mask].mean()) if h2o_mask.sum() > 0 else 1e-5
        feats.append(so2_amp / (h2o_amp + 1e-8))

        # 3. Rayleigh slope (linear fit log-depth vs log-wavelength)
        try:
            log_wl = np.log(wavelengths + 1e-6)
            log_d  = np.log(np.clip(depths, 1e-6, None))
            slope, _ = np.polyfit(log_wl, log_d, 1)
            feats.append(float(np.clip(slope, -5, 5)))
        except Exception:
            feats.append(0.0)

        # 4. DCT coefficients of normalized depths
        d_norm = (depths - depths.mean()) / (depths.std() + 1e-10)
        n_dct = min(self.d - len(feats), 32)
        dct_coeffs = dct(d_norm, type=2, norm="ortho")
        dct_feat = dct_coeffs[:n_dct]
        if len(dct_feat) < n_dct:
            dct_feat = np.pad(dct_feat, (0, n_dct - len(dct_feat)))
        feats.extend(dct_feat.tolist())

        vec = np.array(feats, dtype=float)
        # Pad or truncate to d
        if len(vec) < self.d:
            vec = np.pad(vec, (0, self.d - len(vec)))
        else:
            vec = vec[: self.d]

        # L2-normalize
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)


# -----------------------------------------------------------------------
# Photometric Encoder
# -----------------------------------------------------------------------

class PhotometricEncoder:
    """
    Encodes a transit light curve or photometric time series.
    Features: transit depth, duration, ingress slope, limb-darkening proxy,
              phase-curve amplitude, temporal autocorrelation.
    """

    def __init__(self, d: int = D_EMB):
        self.d = d

    def encode(
        self,
        times: np.ndarray,
        fluxes: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        feats = []

        flux_norm = fluxes / (fluxes.max() + 1e-10)

        # Transit depth (1 - min flux)
        feats.append(float(1.0 - flux_norm.min()))

        # Duration proxy: fraction of points below 99% flux
        feats.append(float((flux_norm < 0.99).mean()))

        # Flux variance
        feats.append(float(flux_norm.var()))

        # RMS scatter (out-of-transit proxy)
        oot = flux_norm[flux_norm > 0.98]
        feats.append(float(oot.std()) if len(oot) > 2 else 0.0)

        # Temporal autocorrelation at lag 1
        if len(flux_norm) > 2:
            ac = float(np.corrcoef(flux_norm[:-1], flux_norm[1:])[0, 1])
            feats.append(ac if np.isfinite(ac) else 0.0)
        else:
            feats.append(0.0)

        # DCT of normalized flux
        n_dct = min(self.d - len(feats), 32)
        dct_c = dct(flux_norm - flux_norm.mean(), type=2, norm="ortho")
        dct_f = dct_c[:n_dct]
        if len(dct_f) < n_dct:
            dct_f = np.pad(dct_f, (0, n_dct - len(dct_f)))
        feats.extend(dct_f.tolist())

        vec = np.array(feats, dtype=float)
        if len(vec) < self.d:
            vec = np.pad(vec, (0, self.d - len(vec)))
        else:
            vec = vec[: self.d]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)


# -----------------------------------------------------------------------
# Model / Retrieval Encoder
# -----------------------------------------------------------------------

class ModelEncoder:
    """
    Encodes an atmospheric model parameter vector into D_EMB-dim space.
    Standardizes parameters and appends χ² residual features.
    """

    PARAM_KEYS = ["T_eq", "log_g", "log_H2O", "log_CO2", "log_SO2",
                  "log_CO", "log_CH4"]
    PARAM_MEANS = [1000.0, 3.0, -3.0, -3.5, -7.0, -3.5, -5.0]
    PARAM_STDS  = [400.0,  0.5,  1.5,  1.5,  2.5,  1.5,  2.0]

    def __init__(self, d: int = D_EMB):
        self.d = d

    def encode(
        self,
        params: Dict,
        chi2: float = 1.0,
        n_wavelengths: int = 150,
    ) -> np.ndarray:
        feats = []
        for key, mu, sig in zip(self.PARAM_KEYS, self.PARAM_MEANS, self.PARAM_STDS):
            val = params.get(key, mu)
            feats.append((val - mu) / sig)

        # χ² and model-family flag
        feats.append(np.log1p(max(chi2, 0.0)))
        feats.append(1.0 if params.get("label", "").startswith("photo") else 0.0)

        # Synthetic spectrum residual fingerprint (DCT of forward model)
        T = params.get("T_eq", 1000.0)
        SO2 = 10 ** params.get("log_SO2", -10.0)
        CO2 = 10 ** params.get("log_CO2", -3.5)
        fingerprint = np.sin(np.linspace(0, np.pi * T / 500.0, 32)) * CO2 * 10
        fingerprint += np.cos(np.linspace(0, np.pi, 32)) * SO2 * 50
        dct_fp = dct(fingerprint, type=2, norm="ortho")[:min(self.d - len(feats), 32)]
        feats.extend(dct_fp.tolist())

        vec = np.array(feats, dtype=float)
        if len(vec) < self.d:
            vec = np.pad(vec, (0, self.d - len(vec)))
        else:
            vec = vec[: self.d]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)


# -----------------------------------------------------------------------
# Literature Encoder
# -----------------------------------------------------------------------

class LiteratureEncoder:
    """
    Encodes a paper abstract/title into D_EMB-dim space via TF-IDF
    over a small astrophysics keyword vocabulary.
    """

    VOCAB = [
        "water", "carbon dioxide", "sulfur dioxide", "photochemistry",
        "equilibrium", "cloud", "haze", "transmission", "emission",
        "retrieval", "atmosphere", "exoplanet", "jwst", "miri", "nirspec",
        "niriss", "nircam", "wasp", "trappist", "temperature", "metallicity",
        "abundance", "biosignature", "habitability", "rayleigh", "scattering",
        "nested sampling", "bayesian", "petitradtrans", "poseidon", "aurora",
        "SO2", "CO2", "H2O", "CH4", "CO", "Na", "K", "disequilibrium",
    ]

    def __init__(self, d: int = D_EMB):
        self.d = d
        # IDF weights (uniform for simplicity; learned from ADS in production)
        self._idf = np.ones(len(self.VOCAB))

    def encode(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        tf = np.array([text_lower.count(w.lower()) for w in self.VOCAB], dtype=float)
        tfidf = tf * self._idf
        # Append bigram-based chemical signal
        so2_flag = float("so2" in text_lower or "sulfur dioxide" in text_lower)
        photo_flag = float("photochem" in text_lower)
        vec = np.concatenate([tfidf, [so2_flag, photo_flag]])
        if len(vec) < self.d:
            vec = np.pad(vec, (0, self.d - len(vec)))
        else:
            vec = vec[: self.d]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)
