from __future__ import annotations

import json
import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _parse_env_val(val: str):
    """Best-effort parse of an env var string: JSON decode, fallback to raw str."""
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError, ValueError):
        return val


class SpectralConfig(BaseModel):
    """SVD of pre-softmax scores matrix S = QK^T."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]


class DegreeNormalizedConfig(BaseModel):
    """SVD of post-softmax degree-normalized operator M = D_Q^{-1/2} A D_K^{-1/2}."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]
    threshold: int = 2048
    block_size: int = 256
    hodge: bool = False
    hodge_target_cv: float = 0.05
    hodge_curl_seed: int = 42


class GlassboxConfig(BaseSettings):
    """Root configuration for the Glassbox observability framework.

    Precedence (highest wins):
      1. Programmatic kwargs
      2. YAML config file (glassbox.yaml)
      3. Legacy GLASSBOX_SVD_* env var migration
      4. Field defaults
    """

    model_config = SettingsConfigDict(
        yaml_file="glassbox.yaml",
        extra="ignore",
        frozen=True,
    )

    spectral: SpectralConfig = SpectralConfig()
    degree_normalized: DegreeNormalizedConfig = DegreeNormalizedConfig()
    output: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_env(cls, data):
        """Map GLASSBOX_SVD_* env vars to new nested structure."""
        if not isinstance(data, dict):
            return data

        legacy_map = {
            "GLASSBOX_SVD_INTERVAL": ("spectral", "interval"),
            "GLASSBOX_SVD_RANK": ("spectral", "rank"),
            "GLASSBOX_SVD_METHOD": ("spectral", "method"),
            "GLASSBOX_SVD_HEADS": ("spectral", "heads"),
            "GLASSBOX_SVD_OUTPUT": (None, "output"),
            "GLASSBOX_SVD_THRESHOLD": ("degree_normalized", "threshold"),
            "GLASSBOX_SVD_BLOCK_SIZE": ("degree_normalized", "block_size"),
            "GLASSBOX_SVD_HODGE": ("degree_normalized", "hodge"),
            "GLASSBOX_SVD_HODGE_TARGET_CV": ("degree_normalized", "hodge_target_cv"),
            "GLASSBOX_SVD_HODGE_CURL_SEED": ("degree_normalized", "hodge_curl_seed"),
        }

        found_legacy = False
        for env_key, (section, field) in legacy_map.items():
            val = os.environ.get(env_key)
            if val is None:
                continue
            found_legacy = True
            parsed = _parse_env_val(val)
            if section is None:
                data.setdefault(field, parsed)
            else:
                section_dict = data.setdefault(section, {})
                if isinstance(section_dict, dict) and field not in section_dict:
                    section_dict[field] = parsed

        # GLASSBOX_SVD_OPERATOR=M -> enable degree_normalized, disable spectral
        op = os.environ.get("GLASSBOX_SVD_OPERATOR")
        if op is not None:
            found_legacy = True
            if op.upper() == "M":
                dn = data.setdefault("degree_normalized", {})
                if isinstance(dn, dict):
                    dn.setdefault("enabled", True)
                    for shared_env, field in [
                        ("GLASSBOX_SVD_INTERVAL", "interval"),
                        ("GLASSBOX_SVD_RANK", "rank"),
                        ("GLASSBOX_SVD_METHOD", "method"),
                        ("GLASSBOX_SVD_HEADS", "heads"),
                    ]:
                        shared_val = os.environ.get(shared_env)
                        if shared_val is not None:
                            dn.setdefault(field, _parse_env_val(shared_val))
                    sp = data.setdefault("spectral", {})
                    if isinstance(sp, dict):
                        sp.setdefault("enabled", False)

        if found_legacy:
            logger.warning(
                "GLASSBOX_SVD_* env vars are deprecated. "
                "Use glassbox.yaml or programmatic config instead."
            )

        return data

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        from pydantic_settings import YamlConfigSettingsSource

        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )
