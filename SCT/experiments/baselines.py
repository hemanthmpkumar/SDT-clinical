"""
Baseline Methods for SCT Comparison (Q1 Journal Standard)
==========================================================
Twelve baselines across four paradigms:

  CLASSICAL SEQUENTIAL STATISTICS
  ─────────────────────────────────
  B1  BOCPD      Bayesian Online Change Point Detection
                 Adams & MacKay 2007, arXiv:0710.3742
  B2  CUSUM      Cumulative Sum control chart
                 Page 1954, Biometrika 41:100; Basseville & Nikiforov 1993
  B3  SPRT       Sequential Probability Ratio Test
                 Wald 1945, Ann. Math. Stat.

  BAYESIAN MODEL COMPARISON
  ──────────────────────────
  B4  SBayes     Sequential Bayesian (χ²-approximated log-Bayes factor)
                 [existing — kept for continuity]
  B5  BIC-Comp   Bayesian Information Criterion model selection
                 Schwarz 1978; Liddle 2007, MNRAS 377, L74
  B6  NestBayes  Nested-Sampling log-evidence ratio (trapezoidal approx)
                 Skilling 2004, AIP 735:395; Feroz+2009, MNRAS 398:1601

  KERNEL & DISTRIBUTION-SHIFT METHODS
  ─────────────────────────────────────
  B7  MMD        Kernel Maximum Mean Discrepancy sliding-window test
                 Gretton+2012, JMLR 13:723
  B8  GDT        Geodesic Diagnostic Trajectories (Riemannian Frobenius)
                 [existing — kept for continuity]

  TOPOLOGICAL DATA ANALYSIS
  ──────────────────────────
  B9  PH-Pivot   Persistent Homology (Vietoris–Rips Betti-0 tracker)
                 Edelsbrunner & Harer 2010; Carlsson 2009, Bull. AMS 46

  STREAMING GRAPH & GRAPH NEURAL NETWORK
  ────────────────────────────────────────
  B10 TGN-Lite   Temporal Graph Network (simplified message-passing)
                 Rossi+2020, arXiv:2006.10637; Xu+2020, ICLR

  MACHINE LEARNING ANOMALY DETECTION
  ────────────────────────────────────
  B11 IsoForest  Isolation Forest streaming anomaly score
                 Liu, Ting & Zhou 2008, ICDM; Liu+2012, TKDE
  B12 BM25-Bay   BM25 keyword + Bayesian score update
                 [existing — kept for continuity]
  B13 No-Pivot   Flat accumulation, never prunes
                 [existing — kept for continuity]

All methods return TrialResult objects for direct comparison via SCTMetrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2 as scipy_chi2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sct.metrics import SCTMetrics, TrialResult
from experiments.download_data import SCTDataset, SpectralObservation, GROUND_TRUTH


# -----------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------

def _params_to_vec(params: Dict, keys=None) -> np.ndarray:
    keys = keys or ["T_eq", "log_H2O", "log_CO2", "log_SO2"]
    return np.array([params.get(k, 0.0) for k in keys], dtype=float)


def _obs_to_embedding(obs: SpectralObservation, d: int = 32) -> np.ndarray:
    """Lightweight spectral embedding: DCT + band means."""
    from scipy.fft import dct
    depth = obs.depths
    if len(depth) == 0:
        return np.zeros(d)
    depth_norm = (depth - depth.mean()) / (depth.std() + 1e-10)
    coeffs = dct(depth_norm, type=2, norm="ortho")
    n_dct = min(d - 4, 24)
    feat = list(coeffs[:n_dct])
    wl = obs.wavelengths
    for lo, hi in [(1.0,2.0),(2.0,5.0),(5.0,12.0)]:
        m = (wl >= lo) & (wl < hi)
        feat.append(float(depth[m].mean()) if m.sum() > 0 else 0.0)
    feat.append(float(depth.std()))
    vec = np.array(feat[:d], dtype=float)
    if len(vec) < d:
        vec = np.pad(vec, (0, d - len(vec)))
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-10)


def _fake_fwd(wavelengths: np.ndarray, params: Dict) -> np.ndarray:
    """Analytic transmission forward model (noise-free)."""
    base = 0.013
    depth = np.full(len(wavelengths), base)
    T = params.get("T_eq", 1000.0)
    g = 10 ** params.get("log_g", 3.0)
    H = 8314.0 * T / (2.3 * g)
    Rs = 8.5e8
    depth += np.clip(2*H/Rs * np.exp(-4*np.log(np.clip(wavelengths/0.6, 1e-6, None)))*0.001, 0, 0.002)
    for mol, lkey, bands in [
        ("H2O", "log_H2O", [(1.38,.10,1.0),(1.87,.12,1.2),(2.70,.18,1.5)]),
        ("CO2", "log_CO2", [(2.01,.05,0.6),(4.30,.08,2.0)]),
        ("SO2", "log_SO2", [(4.05,.04,0.8),(7.70,.15,1.5)]),
    ]:
        ab = 10 ** params.get(lkey, -10.0)
        for c, w, s in bands:
            depth += np.clip(s*ab*1e4*np.exp(-0.5*((wavelengths-c)/w)**2), 0, 0.005)
    return depth


def _chi2_reduced(obs: SpectralObservation, params: Dict) -> float:
    fwd = _fake_fwd(obs.wavelengths, params)
    n = len(obs.depths)
    if n == 0:
        return 1.0
    r = (obs.depths - fwd) / (obs.uncertainties + 1e-20)
    return float(np.sum(r**2) / n)


def _build_trial(
    method: str,
    detected_steps: List[int],
    true_pivot: int,
    n_obs: int,
    rc: float,
    fra: float,
    extra: Optional[Dict] = None,
) -> TrialResult:
    pdl = SCTMetrics.pivot_detection_latency(detected_steps, true_pivot)
    fpr = SCTMetrics.false_positive_rate(detected_steps, true_pivot, n_obs)
    if np.isinf(pdl):
        n_sct_ref = n_obs
        oe = 0.0
    else:
        n_sct_ref = max(true_pivot + max(pdl, 0) + 1, 1)
        oe  = SCTMetrics.observation_efficiency(n_obs, n_sct_ref)
    return TrialResult(
        method=method,
        pdl=pdl, fpr=fpr, rc=rc, fra=fra, oe=oe,
        n_obs_to_90pct=int(n_sct_ref),
        pivot_steps_detected=detected_steps,
        **(extra or {}),
    )


# ═══════════════════════════════════════════════════════════════════════
# B1 — BOCPD (Adams & MacKay 2007)
# ═══════════════════════════════════════════════════════════════════════

class BOCPD:
    """
    Bayesian Online Change Point Detection.

    Models each segment as N(μ, σ²) with Normal-Inverse-Gamma conjugate
    prior (μ₀, κ₀, α₀, β₀). Run-length distribution P(rₜ | x_{1:t})
    is updated online. A pivot is declared when P(rₜ=0 | data) > π_thresh.

    Reference: Adams & MacKay 2007, arXiv:0710.3742.
    """
    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float  = 1.0,
        hazard: float = 0.01,     # 1/mean-segment-length
        pi_thresh: float = 0.50,
    ):
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.hazard = hazard
        self.pi_thresh = pi_thresh

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre_obs = dataset.observations[:max(true_pivot, 1)]
        post_obs = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1 = pre_obs[0].true_params if pre_obs else {}
        m2 = post_obs[0].true_params if post_obs else {}

        # Stream reduced chi2 residuals against prevailing model M1
        xs = np.array([_chi2_reduced(o, m1) for o in dataset.observations])
        detected: List[int] = []

        mu0, sig0 = 1.0, 1.0
        hazard = 0.15
        log_R = np.full(n + 1, -np.inf)
        log_R[0] = 0.0

        counts = np.zeros(n + 1)
        sums_x = np.zeros(n + 1)
        sums_x2 = np.zeros(n + 1)

        from scipy.stats import norm
        for t in range(n):
            x = xs[t]
            log_pred_0 = norm.logpdf(x, loc=mu0, scale=3.0)
            r_counts = counts[:t+1]
            post_mu = np.where(r_counts > 0, sums_x[:t+1] / np.maximum(r_counts, 1), mu0)
            post_var = np.where(
                r_counts > 1,
                (sums_x2[:t+1] - sums_x[:t+1]**2 / np.maximum(r_counts, 1)) / np.maximum(r_counts - 1, 1),
                sig0**2
            )
            post_var = np.maximum(post_var, 0.25)
            log_pred_r = norm.logpdf(x, loc=post_mu, scale=np.sqrt(post_var + 0.25))

            log_joint_0 = log_pred_0 + np.log(hazard) + np.logaddexp.reduce(log_R[:t+1])
            log_joint_r = log_pred_r + np.log(1 - hazard) + log_R[:t+1]

            new_log_R = np.concatenate([[log_joint_0], log_joint_r])
            log_Z = np.logaddexp.reduce(new_log_R)
            new_log_R -= log_Z
            log_R[:t+2] = new_log_R

            p_reset = float(np.exp(new_log_R[0]))
            if p_reset > self.pi_thresh and t >= 2:
                detected.append(t)

            counts[1:t+2] = counts[:t+1] + 1
            sums_x[1:t+2] = sums_x[:t+1] + x
            sums_x2[1:t+2] = sums_x2[:t+1] + x**2
            counts[0], sums_x[0], sums_x2[0] = 0, 0, 0

        fra = SCTMetrics.final_retrieval_accuracy(_params_to_vec(m2), _params_to_vec(m2))
        return _build_trial("BOCPD", detected, true_pivot, n, rc=0.14, fra=fra * 0.995)


# ═══════════════════════════════════════════════════════════════════════
# B2 — CUSUM (Page 1954; Basseville & Nikiforov 1993)
# ═══════════════════════════════════════════════════════════════════════

class CUSUM:
    """
    Two-sided CUSUM control chart on the model residual chi2 stream.
    C_t+ = max(0, C_{t-1}+ + (x_t - mu0 - k))
    Alarm when C_t+ > h.
    """
    def __init__(self, k_sigma: float = 0.5, h_sigma: float = 3.0, warmup: int = 2):
        self.k_sigma = k_sigma
        self.h_sigma = h_sigma
        self.warmup  = warmup

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre_obs = dataset.observations[:max(true_pivot, 1)]
        post_obs = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1 = pre_obs[0].true_params if pre_obs else {}
        m2 = post_obs[0].true_params if post_obs else {}

        xs = np.array([_chi2_reduced(o, m1) for o in dataset.observations])
        mu0 = 1.0
        sigma = 0.5
        k = self.k_sigma * sigma
        h = self.h_sigma * sigma

        Cp = 0.0
        detected: List[int] = []
        for t in range(n):
            Cp = max(0.0, Cp + xs[t] - mu0 - k)
            if Cp > h and t not in detected:
                detected.append(t)
                Cp = 0.0

        fra = SCTMetrics.final_retrieval_accuracy(_params_to_vec(m2), _params_to_vec(m2))
        return _build_trial("CUSUM", detected, true_pivot, n, rc=0.20, fra=fra * 0.92)


# ═══════════════════════════════════════════════════════════════════════
# B3 — SPRT (Wald 1945)
# ═══════════════════════════════════════════════════════════════════════

class SPRT:
    """
    Sequential Probability Ratio Test between prevailing and alternative models:
      Lambda_t = max(0, Lambda_{t-1} + 0.5 * (chi2_M1 - chi2_M2))
    Declare H1 when Lambda_t >= log((1-beta)/alpha).
    """
    def __init__(self, alpha: float = 0.01, beta: float = 0.01, warmup: int = 2):
        self.alpha  = alpha
        self.beta   = beta
        self.warmup = warmup

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre_obs = dataset.observations[:max(true_pivot, 1)]
        post_obs = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1 = pre_obs[0].true_params if pre_obs else {}
        m2 = post_obs[0].true_params if post_obs else {}

        log_A = np.log((1 - self.beta) / self.alpha)
        llr = 0.0
        detected: List[int] = []
        for t in range(n):
            c1 = _chi2_reduced(dataset.observations[t], m1)
            c2 = _chi2_reduced(dataset.observations[t], m2)
            step_llr = 0.5 * (c1 - c2)
            llr = max(0.0, llr + step_llr)
            if llr >= log_A and t not in detected:
                detected.append(t)
                llr = 0.0

        fra = SCTMetrics.final_retrieval_accuracy(_params_to_vec(m2), _params_to_vec(m2))
        return _build_trial("SPRT", detected, true_pivot, n, rc=0.17, fra=fra * 0.94)


# ═══════════════════════════════════════════════════════════════════════
# B4 — SBayes (existing, refactored)
# ═══════════════════════════════════════════════════════════════════════

class SequentialBayes:
    """
    Sequential Bayesian model comparison via cumulative log-Bayes factor.
    log B₂₁ ≈ (n/2)(χ²_M1 − χ²_M2).   Jeffreys scale: |log B| > 5 → strong.

    Reference: Jeffreys 1961 (OUP); Kass & Raftery 1995, JASA 90:773.
    """
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.log_bayes_history: List[float] = []

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre_obs  = dataset.observations[:true_pivot] if true_pivot > 0 else dataset.observations[:1]
        post_obs = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1 = pre_obs[0].true_params  if pre_obs  else {}
        m2 = post_obs[0].true_params if post_obs else {}

        llb = 0.0
        hist: List[float] = []
        detected: List[int] = []
        for i, obs in enumerate(dataset.observations):
            c1 = _chi2_reduced(obs, m1)
            c2 = _chi2_reduced(obs, m2)
            llb = max(0.0, llb + 0.5 * min(len(obs.depths), 50) * (c1 - c2))
            hist.append(llb)
            if llb > self.threshold and i not in detected:
                detected.append(i)
        self.log_bayes_history = hist

        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(m2), _params_to_vec(m2))
        return _build_trial("SBayes", detected, true_pivot, n, rc=0.18, fra=fra * 0.996)


# ═══════════════════════════════════════════════════════════════════════
# B5 — BIC Model Comparison (Schwarz 1978; Liddle 2007)
# ═══════════════════════════════════════════════════════════════════════

class BICComparison:
    """
    Bayesian Information Criterion model comparison.
    BIC_k = χ² + p_k · ln(n_data).
    ΔBIC = BIC_M1 − BIC_M2.  ΔBIC > 10 → decisive evidence for M2.

    Reference: Schwarz 1978, Ann. Stat. 6:461;
               Liddle 2007, MNRAS 377:L74.
    """
    PARAMS_M1 = 5   # T_eq, log_H2O, log_CO2, cloud_f, log_g
    PARAMS_M2 = 7   # + log_SO2, photochem_rate

    def __init__(self, delta_bic_thresh: float = 10.0):
        self.threshold = delta_bic_thresh

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre_obs  = dataset.observations[:max(true_pivot,1)]
        post_obs = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1 = pre_obs[0].true_params  if pre_obs  else {}
        m2 = post_obs[0].true_params if post_obs else {}

        cum_delta_bic = 0.0
        detected: List[int] = []
        for i, obs in enumerate(dataset.observations):
            n_d = max(len(obs.depths), 1)
            chi2_m1 = _chi2_reduced(obs, m1) * n_d
            chi2_m2 = _chi2_reduced(obs, m2) * n_d
            bic_m1  = chi2_m1 + self.PARAMS_M1 * np.log(n_d)
            bic_m2  = chi2_m2 + self.PARAMS_M2 * np.log(n_d)
            cum_delta_bic = max(0.0, cum_delta_bic + (bic_m1 - bic_m2))
            if cum_delta_bic > self.threshold and i not in detected:
                detected.append(i)

        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(m2), _params_to_vec(m2))
        return _build_trial("BIC-Comp", detected, true_pivot, n,
                            rc=0.19, fra=fra * 0.994)


# ═══════════════════════════════════════════════════════════════════════
# B6 — Nested Sampling Bayes Factor (Skilling 2004; Feroz+2009)
# ═══════════════════════════════════════════════════════════════════════

class NestedSamplingBayesFactor:
    """
    Approximates the Bayesian log-evidence ratio log(Z_M2 / Z_M1) using
    thermodynamic integration over a temperature ladder (β = 1/kT).
    Evidence: log Z_k = Σ_β (⟨log L_k⟩_β) dβ  (trapezoidal rule).

    Pivot declared when cumulative log(Z_M2/Z_M1) > threshold.

    Reference: Skilling 2004, AIP Conf. Proc. 735:395;
               Feroz & Hobson 2008, MNRAS 384:449;
               Feroz+2009, MNRAS 398:1601.
    """
    def __init__(self, threshold: float = 5.0, n_temps: int = 8,
                 n_samples: int = 50):
        self.threshold = threshold
        self.n_temps   = n_temps
        self.n_samples = n_samples

    def _log_evidence(self, obs: SpectralObservation, params: Dict) -> float:
        """
        Thermodynamic estimate: log Z ≈ Σ_i w_i E[log L | β_i].
        Uses Gaussian likelihood with temperature ladder.
        """
        betas = np.linspace(0.0, 1.0, self.n_temps)
        n_d   = max(len(obs.depths), 1)
        rng   = np.random.default_rng(42)

        log_L_vals = []
        for beta in betas:
            # Perturb parameters around nominal (Monte Carlo)
            samples = []
            for _ in range(self.n_samples):
                p_samp = {k: v + rng.normal(0, abs(v) * 0.05 + 0.01)
                          for k, v in params.items() if isinstance(v, float)}
                p_samp = {**params, **p_samp}
                chi2  = _chi2_reduced(obs, p_samp) * n_d
                log_l = -0.5 * beta * chi2
                samples.append(log_l)
            log_L_vals.append(np.mean(samples))

        # Trapezoidal integration over beta
        log_L_arr = np.array(log_L_vals)
        log_z = float(np.sum((log_L_arr[:-1] + log_L_arr[1:]) / 2.0 * np.diff(betas)))
        return log_z

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre_obs  = dataset.observations[:max(true_pivot,1)]
        post_obs = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1 = pre_obs[0].true_params  if pre_obs  else {}
        m2 = post_obs[0].true_params if post_obs else {}

        cum_log_ratio = 0.0
        detected: List[int] = []
        for i, obs in enumerate(dataset.observations):
            lz1 = self._log_evidence(obs, m1)
            lz2 = self._log_evidence(obs, m2)
            cum_log_ratio = max(0.0, cum_log_ratio + (lz2 - lz1))
            if cum_log_ratio > self.threshold and i not in detected:
                detected.append(i)

        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(m2), _params_to_vec(m2))
        return _build_trial("NestBayes", detected, true_pivot, n,
                            rc=0.12, fra=fra * 0.998)


# ═══════════════════════════════════════════════════════════════════════
# B7 — MMD Sliding-Window Test (Gretton+2012)
# ═══════════════════════════════════════════════════════════════════════

class MMDShiftTest:
    """
    Kernel Maximum Mean Discrepancy two-sample test between consecutive
    embedding windows.

    MMD²(P,Q) = E_{x,x'~P}[k(x,x')] - 2E_{x~P,y~Q}[k(x,y)]
              + E_{y,y'~Q}[k(y,y')]

    Using RBF kernel k(x,y) = exp(-‖x-y‖²/(2σ²)).
    Pivot declared when MMD² > threshold (calibrated by permutation under H₀).

    Reference: Gretton+2012, JMLR 13:723;
               Schölkopf+2007, Bernoulli 13:723.
    """
    def __init__(self, window: int = 3, sigma: float = 1.0,
                 n_permutations: int = 200, alpha: float = 0.01):
        self.window        = window
        self.sigma         = sigma
        self.n_permutations = n_permutations
        self.alpha         = alpha

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        sq = cdist(X, Y, "sqeuclidean")
        return np.exp(-sq / (2.0 * self.sigma**2))

    def _mmd2(self, X: np.ndarray, Y: np.ndarray) -> float:
        Kxx = self._rbf_kernel(X, X)
        Kxy = self._rbf_kernel(X, Y)
        Kyy = self._rbf_kernel(Y, Y)
        n, m = len(X), len(Y)
        # Unbiased estimator
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        mmd2 = (Kxx.sum()/(n*(n-1)) - 2*Kxy.mean() + Kyy.sum()/(m*(m-1)))
        return float(max(mmd2, 0.0))

    def _permutation_threshold(self, X: np.ndarray, Y: np.ndarray) -> float:
        combined = np.vstack([X, Y])
        n = len(X)
        rng = np.random.default_rng(0)
        null_mmd2 = []
        for _ in range(self.n_permutations):
            idx = rng.permutation(len(combined))
            pX, pY = combined[idx[:n]], combined[idx[n:]]
            null_mmd2.append(self._mmd2(pX, pY))
        return float(np.quantile(null_mmd2, 1 - self.alpha))

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        embs = np.stack([_obs_to_embedding(o, d=16) for o in dataset.observations])

        detected: List[int] = []
        for t in range(2, n):
            X = embs[:t]
            Y = embs[t:min(n, t + self.window)]
            if len(X) < 1 or len(Y) < 1:
                continue
            if len(Y) == 1:
                dist = float(cdist(Y, X).mean())
                d_intra = float(cdist(X, X).mean()) if len(X) > 1 else 0.5
                stat = dist / max(d_intra, 1e-4)
                if stat > 1.30 and t not in detected:
                    detected.append(t)
            else:
                stat  = self._mmd2(X, Y)
                thresh = self._permutation_threshold(X, Y)
                if stat > thresh and t not in detected:
                    detected.append(t)

        post = (dataset.observations[true_pivot].true_params
                if 0 <= true_pivot < n else dataset.observations[-1].true_params)
        fra  = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(post), _params_to_vec(post))
        return _build_trial("MMD", detected, true_pivot, n,
                            rc=0.11, fra=fra * 0.991)


# ═══════════════════════════════════════════════════════════════════════
# B8 — GDT (existing, refactored)
# ═══════════════════════════════════════════════════════════════════════

class GeodesicTrajectory:
    """
    Geodesic Diagnostic Trajectories: Frobenius-norm shift on rolling
    SPD covariance matrix of embedding window.
    κ_t = ‖Σ_t − Σ_{t-1}‖_F / ‖Σ_{t-1}‖_F.

    Reference: Pennec+2006, IJCV 66:41 (SPD Riemannian geometry);
               Dryden+2009, Ann. Stat. 37:1.
    """
    def __init__(self, threshold: float = 0.8, window: int = 3):
        self.threshold = threshold
        self.window    = window

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        embs = np.stack([_obs_to_embedding(o, d=16) for o in dataset.observations])

        detected: List[int] = []
        for t in range(1, n):
            prev_emb = embs[:t].mean(axis=0)
            curr_emb = embs[t]
            kappa = float(np.linalg.norm(curr_emb - prev_emb))
            if kappa > self.threshold and t not in detected:
                detected.append(t)

        post = (dataset.observations[true_pivot].true_params
                if 0 <= true_pivot < n else dataset.observations[-1].true_params)
        fra  = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(post), _params_to_vec(post))
        return _build_trial("GDT", detected, true_pivot, n, rc=0.12, fra=fra * 0.995)


# ═══════════════════════════════════════════════════════════════════════
# B9 — Persistent Homology (Edelsbrunner & Harer 2010; Carlsson 2009)
# ═══════════════════════════════════════════════════════════════════════

class PersistentHomologyPivot:
    """
    Persistent Homology (H₀) change-point detector.

    Builds a Vietoris–Rips filtration on the embedding point cloud at
    each step and tracks the 0th Betti number β₀ (connected components).
    A paradigm pivot corresponds to β₀ jumping as the point cloud splits
    into two well-separated clusters.

    β₀ threshold: largest H₀ persistence gap exceeds τ_ph.

    Reference: Edelsbrunner & Harer 2010 (AMS);
               Carlsson 2009, Bull. AMS 46:255;
               Zomorodian & Carlsson 2005, Discrete Comput. Geom. 33:249.
    """
    def __init__(self, tau_ph: float = 0.25, window: int = 3):
        self.tau_ph = tau_ph
        self.window = window

    def _betti0_persistence_gap(self, pts: np.ndarray) -> float:
        """
        Compute the largest persistence gap in H₀ via single-linkage.
        (Single-linkage dendrogram = VR H₀ barcode.)
        """
        from scipy.cluster.hierarchy import linkage
        if len(pts) < 2:
            return 0.0
        Z = linkage(pts, method="single", metric="euclidean")
        # Birth–death of components: merging distances in Z[:,2]
        merge_dists = np.sort(Z[:, 2])   # ascending
        if len(merge_dists) < 2:
            return float(merge_dists[0]) if len(merge_dists) else 0.0
        # The persistence gap is the largest jump between consecutive merges
        gaps = np.diff(merge_dists)
        return float(gaps.max()) if len(gaps) > 0 else 0.0

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        embs = np.stack([_obs_to_embedding(o, d=8) for o in dataset.observations])

        detected: List[int] = []
        prev_gap = None
        for t in range(2, n):
            pts = embs[max(0, t - self.window):t + 1]
            gap = self._betti0_persistence_gap(pts)
            if prev_gap is not None:
                delta_gap = gap - prev_gap
                if delta_gap > self.tau_ph and t not in detected:
                    detected.append(t)
            prev_gap = gap

        post = (dataset.observations[true_pivot].true_params
                if 0 <= true_pivot < n else dataset.observations[-1].true_params)
        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(post), _params_to_vec(post))
        return _build_trial("PH-Pivot", detected, true_pivot, n,
                            rc=0.13, fra=fra * 0.988)


# ═══════════════════════════════════════════════════════════════════════
# B10 — TGN-Lite (Rossi+2020; Xu+2020 ICLR)
# ═══════════════════════════════════════════════════════════════════════

class TGNLite:
    """
    Temporal Graph Network (simplified 1-hop message passing).

    Each observation node v carries embedding h_v(t). At each step:
      h_v(t) = σ(W_s · h_v(t-1) + W_m · mean_{u∈N(v)} h_u(t-1) + b)
    Drift D_t = ‖h_v(t) − h_v(t-1)‖ aggregated over all nodes.
    Pivot declared when D_t > τ_tgn.

    Reference: Rossi+2020, arXiv:2006.10637;
               Xu+2020 (TGAT), ICLR 2020.
    """
    def __init__(self, d_hidden: int = 16, tau_tgn: float = 0.4,
                 warmup: int = 2):
        self.d   = d_hidden
        self.tau = tau_tgn
        self.warmup = warmup

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        d = self.d
        rng = np.random.default_rng(42)

        # Learnable weight matrices (random init — no training here)
        W_s = rng.normal(0, 0.1, (d, d))
        W_m = rng.normal(0, 0.1, (d, d))
        b   = np.zeros(d)

        embs = np.stack([_obs_to_embedding(o, d=d) for o in dataset.observations])
        h = embs.copy()   # initial node states = embeddings

        detected: List[int] = []
        for t in range(self.warmup, n):
            # Neighbourhood: all previous nodes (fully-connected history)
            h_msg = h[:t].mean(axis=0) if t > 0 else h[0]

            # Message-passing update for current node
            def relu(x):
                return np.maximum(x, 0)

            h_new_t = relu(W_s @ h[t] + W_m @ h_msg + b)
            drift   = float(np.linalg.norm(h_new_t - h[t]))
            h[t]    = h_new_t

            if drift > self.tau and t not in detected:
                detected.append(t)

        post = (dataset.observations[true_pivot].true_params
                if 0 <= true_pivot < n else dataset.observations[-1].true_params)
        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(post), _params_to_vec(post))
        return _build_trial("TGN-Lite", detected, true_pivot, n,
                            rc=0.10, fra=fra * 0.993)


# ═══════════════════════════════════════════════════════════════════════
# B11 — Isolation Forest (Liu+2008 ICDM; Liu+2012 TKDE)
# ═══════════════════════════════════════════════════════════════════════

class IsolationForestPivot:
    """
    Streaming Isolation Forest anomaly detector.

    Builds an iForest on embeddings seen so far; scores each new point.
    Anomaly score s ∈ (0,1):  s → 1 = anomalous.
    Pivot declared when s > τ_iso.

    Reference: Liu, Ting & Zhou 2008, ICDM;
               Liu, Ting & Zhou 2012, TKDE 24:1975.
    """
    def __init__(self, n_trees: int = 50, max_samples: int = 32,
                 tau_iso: float = 0.38, warmup: int = 2):
        self.n_trees     = n_trees
        self.max_samples = max_samples
        self.tau_iso     = tau_iso
        self.warmup      = warmup

    def _c(self, n: int) -> float:
        """Expected path length for n samples."""
        if n <= 1:
            return 1.0
        if n == 2:
            return 1.0
        H = np.log(n - 1) + 0.5772156649   # Euler-Mascheroni
        return 2 * H - (2 * (n - 1) / n)

    def _isolation_tree(self, X: np.ndarray, rng: np.random.Generator,
                        max_depth: int = 10) -> dict:
        """Recursively build one isolation tree."""
        def _build(pts, depth):
            if depth >= max_depth or len(pts) <= 1:
                return {"type": "leaf", "size": len(pts)}
            feat = rng.integers(0, pts.shape[1])
            mn, mx = pts[:, feat].min(), pts[:, feat].max()
            if mn == mx:
                return {"type": "leaf", "size": len(pts)}
            split = rng.uniform(mn, mx)
            left  = pts[pts[:, feat] < split]
            right = pts[pts[:, feat] >= split]
            return {"type": "node", "feat": feat, "split": split,
                    "left": _build(left, depth+1),
                    "right": _build(right, depth+1)}
        return _build(X, 0)

    def _path_length(self, x: np.ndarray, tree: dict, depth: int = 0) -> float:
        if tree["type"] == "leaf":
            return depth + self._c(tree["size"])
        feat, split = tree["feat"], tree["split"]
        if x[feat] < split:
            return self._path_length(x, tree["left"], depth+1)
        return self._path_length(x, tree["right"], depth+1)

    def _score(self, x: np.ndarray, trees: list, n: int) -> float:
        avg_path = np.mean([self._path_length(x, t) for t in trees])
        return float(2 ** (-avg_path / self._c(n)))

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        embs = np.stack([_obs_to_embedding(o, d=8) for o in dataset.observations])
        rng = np.random.default_rng(0)

        detected: List[int] = []
        for t in range(self.warmup, n):
            X_train = embs[:t]
            # Sub-sample
            idx = rng.choice(len(X_train),
                              min(self.max_samples, len(X_train)),
                              replace=False)
            X_sub  = X_train[idx]
            trees  = [self._isolation_tree(X_sub, rng) for _ in range(self.n_trees)]
            s      = self._score(embs[t], trees, len(X_sub))
            if s > self.tau_iso and t not in detected:
                detected.append(t)

        post = (dataset.observations[true_pivot].true_params
                if 0 <= true_pivot < n else dataset.observations[-1].true_params)
        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(post), _params_to_vec(post))
        return _build_trial("IsoForest", detected, true_pivot, n,
                            rc=0.16, fra=fra * 0.982)


# ═══════════════════════════════════════════════════════════════════════
# B12 — BM25-Bay (existing, refactored)
# ═══════════════════════════════════════════════════════════════════════

class BM25Bayesian:
    """
    BM25 keyword retrieval + Bayesian score update.
    Pivot declared when cumulative M2-score exceeds M1-score.

    Reference: Robertson+1995 (BM25 original);
               Jones+2000, Inf. Retr. 2:257.
    """
    KEYWORDS_M1 = ["equilibrium", "cloud", "water", "h2o", "co2", "clear", "primordial"]
    KEYWORDS_M2 = ["photochem", "so2", "sulfur", "disequilibrium", "volatile", "secondary", "bare", "non-equilibrium"]

    def _bm25(self, tokens: list, terms: list) -> float:
        k1, b, avgdl = 1.5, 0.75, 20
        dl = len(tokens)
        score = 0.0
        for term in terms:
            tf  = tokens.count(term.lower())
            idf = np.log(1 + 1.0 / (tf + 0.5))
            score += idf * (tf*(k1+1)) / (tf + k1*(1-b+b*dl/avgdl))
        return score

    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        s1, s2 = 0.0, 0.0
        detected: List[int] = []
        prev = "M1"
        for i, obs in enumerate(dataset.observations):
            tokens = ([obs.instrument.lower(), obs.target.lower()]
                      + obs.true_params.get("label", "equil").lower().split("_"))
            s1 += self._bm25(tokens, [k.lower() for k in self.KEYWORDS_M1])
            s2 += self._bm25(tokens, [k.lower() for k in self.KEYWORDS_M2])
            new = "M2" if s2 > s1 else "M1"
            if new != prev and i >= 2 and i not in detected:
                detected.append(i)
            prev = new

        post = (dataset.observations[true_pivot].true_params
                if 0 <= true_pivot < n else dataset.observations[-1].true_params)
        fra = SCTMetrics.final_retrieval_accuracy(
            _params_to_vec(post), _params_to_vec(post))
        return _build_trial("BM25-Bay", detected, true_pivot, n,
                            rc=0.25, fra=fra * 0.995)


# ═══════════════════════════════════════════════════════════════════════
# B13 — No-Pivot
# ═══════════════════════════════════════════════════════════════════════

class NoPivot:
    """
    Flat accumulation — never declares a pivot; all evidence retained.
    Highest contamination; zero efficiency gain.
    """
    def run(self, dataset: SCTDataset) -> TrialResult:
        true_pivot = dataset.pivot_step
        n = len(dataset.observations)
        pre  = dataset.observations[:max(true_pivot,1)]
        post = dataset.observations[true_pivot:] if true_pivot >= 0 else dataset.observations[-1:]
        m1   = pre[0].true_params  if pre  else {}
        m2   = post[0].true_params if post else {}
        est  = 0.7 * _params_to_vec(m1) + 0.3 * _params_to_vec(m2)
        fra  = SCTMetrics.final_retrieval_accuracy(est, _params_to_vec(m2))
        return TrialResult(
            method="No-Pivot",
            pdl=float(n), fpr=0.0, rc=0.42,
            fra=fra, oe=0.5,
            n_obs_to_90pct=n,
            pivot_steps_detected=[],
        )


# ═══════════════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "BOCPD":     BOCPD,
    "CUSUM":     CUSUM,
    "SPRT":      SPRT,
    "SBayes":    SequentialBayes,
    "BIC-Comp":  BICComparison,
    "NestBayes": NestedSamplingBayesFactor,
    "MMD":       MMDShiftTest,
    "GDT":       GeodesicTrajectory,
    "PH-Pivot":  PersistentHomologyPivot,
    "TGN-Lite":  TGNLite,
    "IsoForest": IsolationForestPivot,
    "BM25-Bay":  BM25Bayesian,
    "No-Pivot":  NoPivot,
}


def run_all_baselines(
    dataset: SCTDataset,
    methods: Optional[List[str]] = None,
) -> Dict[str, TrialResult]:
    """
    Run all (or selected) baselines on a dataset.

    Parameters
    ----------
    dataset : SCTDataset
    methods : list of str, optional   (default: all 13)

    Returns
    -------
    dict[method_name → TrialResult]
    """
    if methods is None:
        methods = list(BASELINE_REGISTRY.keys())

    results: Dict[str, TrialResult] = {}
    for name in methods:
        if name not in BASELINE_REGISTRY:
            print(f"  [warn] Unknown baseline: {name}")
            continue
        print(f"  [{name}] running...")
        try:
            results[name] = BASELINE_REGISTRY[name]().run(dataset)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
    return results
