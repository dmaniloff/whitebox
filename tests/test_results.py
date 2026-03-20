import math

import pytest

from glassbox.results import (
    SPECTRAL_FEATURE_NAMES,
    DegreeNormalizedFeatures,
    SVDSnapshot,
    ScoresMatrixFeatures,
    _spectral_from_svs,
)


# ── _spectral_from_svs ────────────────────────────────────────────────────


class TestSpectralFromSvs:
    def test_normal(self):
        result = _spectral_from_svs([10.0, 5.0, 2.0])
        assert result["sv1"] == 10.0
        assert result["sv_ratio"] == pytest.approx(2.0)
        assert result["sv_entropy"] is not None
        assert result["sv_entropy"] > 0

    def test_single_value(self):
        result = _spectral_from_svs([7.0])
        assert result["sv1"] == 7.0
        assert result["sv_ratio"] is None
        # entropy of single value: -1.0 * log(1.0 + 1e-12) ≈ 0
        assert result["sv_entropy"] == pytest.approx(0.0, abs=1e-6)

    def test_empty(self):
        result = _spectral_from_svs([])
        assert result == {"sv1": None, "sv_ratio": None, "sv_entropy": None}

    def test_all_zeros(self):
        result = _spectral_from_svs([0.0, 0.0])
        assert result["sv1"] == 0.0
        assert result["sv_ratio"] is None  # division by zero guard
        assert result["sv_entropy"] is None  # total == 0

    def test_second_sv_zero(self):
        result = _spectral_from_svs([5.0, 0.0])
        assert result["sv1"] == 5.0
        assert result["sv_ratio"] is None


# ── ScoresMatrixFeatures ──────────────────────────────────────────────────


class TestScoresMatrixFeatures:
    def test_from_singular_values(self):
        f = ScoresMatrixFeatures.from_singular_values([429.6, 59.0, 41.9])
        assert f.sv1 == 429.6
        assert f.sv_ratio == pytest.approx(429.6 / 59.0)
        assert f.sv_entropy is not None

    def test_frozen(self):
        f = ScoresMatrixFeatures(sv1=1.0)
        with pytest.raises(Exception):
            f.sv1 = 2.0


# ── DegreeNormalizedFeatures ──────────────────────────────────────────────


class TestDegreeNormalizedFeatures:
    def test_spectral_only(self):
        f = DegreeNormalizedFeatures.from_singular_values([1.0, 0.5])
        assert f.sv1 == 1.0
        assert f.sv_ratio == pytest.approx(2.0)
        assert f.phi_hat is None

    def test_with_routing(self):
        routing = {"phi_hat": 0.31, "G": 0.15, "curl_ratio": 0.42}
        f = DegreeNormalizedFeatures.from_singular_values([1.0, 0.5], routing=routing)
        assert f.sv1 == 1.0
        assert f.phi_hat == 0.31
        assert f.G == 0.15
        assert f.curl_ratio == 0.42
        assert f.sigma2 is None  # not in routing dict


# ── SVDSnapshot ───────────────────────────────────────────────────────────


class TestSVDSnapshot:
    def _make_snapshot(self, **overrides):
        defaults = {
            "feature_group": "scores_matrix",
            "request_id": 0,
            "layer": "model.layers.0.self_attn",
            "layer_idx": 0,
            "head": 0,
            "step": 32,
            "L": 128,
            "singular_values": [10.0, 5.0, 2.0],
            "features": ScoresMatrixFeatures.from_singular_values([10.0, 5.0, 2.0]),
        }
        defaults.update(overrides)
        return SVDSnapshot(**defaults)

    def test_construction(self):
        snap = self._make_snapshot()
        assert snap.feature_group == "scores_matrix"
        assert snap.features.sv1 == 10.0

    def test_model_dump_excludes_none(self):
        snap = self._make_snapshot()
        d = snap.model_dump(exclude_none=True)
        assert "tier" not in d
        assert d["feature_group"] == "scores_matrix"
        assert d["features"]["sv1"] == 10.0

    def test_round_trip(self):
        snap = self._make_snapshot()
        d = snap.model_dump(exclude_none=True)
        restored = SVDSnapshot.from_jsonl_row(d)
        assert restored.features.sv1 == snap.features.sv1
        assert restored.features.sv_ratio == snap.features.sv_ratio

    def test_degree_normalized_round_trip(self):
        routing = {"phi_hat": 0.3, "G": 0.15}
        features = DegreeNormalizedFeatures.from_singular_values([1.0, 0.5], routing=routing)
        snap = self._make_snapshot(
            feature_group="degree_normalized_matrix",
            tier="materialized",
            features=features,
        )
        d = snap.model_dump(exclude_none=True)
        assert d["tier"] == "materialized"
        assert d["features"]["phi_hat"] == 0.3
        restored = SVDSnapshot.from_jsonl_row(d)
        assert restored.features.phi_hat == 0.3
        assert restored.features.sv_ratio == pytest.approx(2.0)


# ── SPECTRAL_FEATURE_NAMES ────────────────────────────────────────────────


def test_spectral_feature_names():
    assert SPECTRAL_FEATURE_NAMES == ["sv_ratio", "sv1", "sv_entropy"]
