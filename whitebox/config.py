from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MODEL = "facebook/opt-125m"
    model: str = "meta-llama/Meta-Llama-3-8B"
    hf_token: SecretStr


config = Settings()
