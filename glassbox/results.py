"""Pydantic models for SVD result emission and JSONL parsing.

Provides a single source of truth for derived spectral features
(sv1, sv_ratio, sv_entropy) and structured result types.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

SPECTRAL_FEATURE_NAMES = ["sv_ratio", "sv1", "sv_entropy"]


def _spectral_from_svs(svs: list[float]) -> dict[str, float | None]:
    """Compute spectral features from a list of singular values."""
    if not svs:
        return {"sv1": None, "sv_ratio": None, "sv_entropy": None}
    sv1 = svs[0]
    sv_ratio = svs[0] / svs[1] if len(svs) >= 2 and svs[1] > 0 else None
    total = sum(svs)
    if total > 0:
        ps = [s / total for s in svs]
        sv_entropy = -sum(p * math.log(p + 1e-12) for p in ps)
    else:
        sv_entropy = None
    return {"sv1": sv1, "sv_ratio": sv_ratio, "sv_entropy": sv_entropy}


class ScoresMatrixFeatures(BaseModel):
    """Features from SVD of pre-softmax scores matrix S = QK^T."""

    model_config = ConfigDict(frozen=True)

    # Spectral (from singular values)
    sv1: float | None = Field(None, description="Leading singular value of S.")
    sv_ratio: float | None = Field(None, description="sigma1/sigma2 ratio.")
    sv_entropy: float | None = Field(None, description="Entropy of normalized singular value distribution.")

    @classmethod
    def from_singular_values(cls, svs: list[float]) -> ScoresMatrixFeatures:
        return cls(**_spectral_from_svs(svs))


class DegreeNormalizedFeatures(BaseModel):
    """Features from SVD + Hodge decomposition of degree-normalized M."""

    model_config = ConfigDict(frozen=True)

    # Spectral (from singular values)
    sv1: float | None = Field(None, description="Leading singular value of M.")
    sv_ratio: float | None = Field(None, description="sigma1/sigma2 ratio.")
    sv_entropy: float | None = Field(None, description="Entropy of normalized singular value distribution.")

    # Routing (from Hodge decomposition)
    phi_hat: float | None = Field(None, description="Conductance (1 - sigma2). High = bottlenecked through one mode.")
    sigma2: float | None = Field(None, description="Second singular value of M.")
    G: float | None = Field(None, description="Total asymmetry: ||M_asym||_F / ||M||_F.")
    Gamma: float | None = Field(None, description="Gradient coefficient: potential-driven portion of asymmetry.")
    C: float | None = Field(None, description="Curl coefficient: circulatory portion of asymmetry (triangle-sampled).")
    curl_ratio: float | None = Field(None, description="C / (G + eps). Share of asymmetry that is circulatory.")
    sigma2_asym: float | None = Field(None, description="Second singular value of M_asym.")
    commutator_norm: float | None = Field(None, description="||[M_sym, M_asym]||_F / ||M||_F. Entanglement of symmetric and antisymmetric parts.")

    @classmethod
    def from_singular_values(
        cls,
        svs: list[float],
        routing: dict | None = None,
    ) -> DegreeNormalizedFeatures:
        kwargs = _spectral_from_svs(svs)
        if routing:
            kwargs.update(routing)
        return cls(**kwargs)


class SVDSnapshot(BaseModel):
    """One SVD observation emitted per (request, layer, head, step)."""

    model_config = ConfigDict(frozen=True)

    feature_group: str  # "scores_matrix" | "degree_normalized_matrix"
    request_id: int
    layer: str
    layer_idx: int | None
    head: int
    step: int
    L: int
    singular_values: list[float]
    tier: str | None = None  # "materialized" | "matrix_free"
    features: ScoresMatrixFeatures | DegreeNormalizedFeatures

    @classmethod
    def from_jsonl_row(cls, raw: dict) -> SVDSnapshot:
        """Deserialize a JSONL row, discriminating features by feature_group."""
        d = dict(raw)
        fg = d["feature_group"]
        feat_raw = d["features"]
        if isinstance(feat_raw, dict):
            if fg == "degree_normalized_matrix":
                d["features"] = DegreeNormalizedFeatures(**feat_raw)
            else:
                d["features"] = ScoresMatrixFeatures(**feat_raw)
        return cls(**d)
