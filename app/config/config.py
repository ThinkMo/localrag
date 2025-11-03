import os
import logging
import dataclasses
from typing import Any, Optional

from langchain_core import runnables


VECTOR_STORE_URI = "./milvus_local.db"
DATABASE_URL = "sqlite+aiosqlite:///localrag.db"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


@dataclasses.dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the system."""
    agent_url: str = f"http://localhost:8000/a2a"
    database_url: str = DATABASE_URL
    vector_store_uri: str = VECTOR_STORE_URI
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_retries: int | None = None
    logprobs: bool | None = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[runnables.RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in dataclasses.fields(cls)
            if f.init
        }
        logging.info("Configuration values: %s", values)

        return cls(**{k: v for k, v in values.items() if v})


def _parse_bool(value):
    """Parse boolean value from string."""
    return value.lower() == "true" if value else False


def _parse_int(value, var_name):
    """Parse integer value with error handling."""
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid integer value for {var_name}: {value}") from None


def _parse_headers(value):
    """Parse headers from comma-separated string."""
    try:
        return [tuple(h.split(":", 1)) for h in value.split(",") if ":" in h]
    except Exception:
        raise ValueError(f"Invalid headers format: {value}") from None


def load_uvicorn_config(args=None):
    """
    Load Uvicorn configuration from environment variables and CLI args.
    Returns a dict suitable for passing to uvicorn.Config.
    """
    config_kwargs = {
        "app": "app.app:app",
        "host": os.getenv("UVICORN_HOST", "0.0.0.0"),
        "port": int(os.getenv("UVICORN_PORT", 8000)),
        "log_level": os.getenv("UVICORN_LOG_LEVEL", "info"),
        "reload": args.reload if args else False,
        "reload_dirs": ["app"] if (args and args.reload) else None,
    }

    # Configuration mapping for advanced options
    config_mapping = {
        "UVICORN_PROXY_HEADERS": ("proxy_headers", _parse_bool),
        "UVICORN_FORWARDED_ALLOW_IPS": ("forwarded_allow_ips", str),
        "UVICORN_WORKERS": ("workers", lambda x: _parse_int(x, "UVICORN_WORKERS")),
        "UVICORN_ACCESS_LOG": ("access_log", _parse_bool),
        "UVICORN_LOOP": ("loop", str),
        "UVICORN_HTTP": ("http", str),
        "UVICORN_WS": ("ws", str),
        "UVICORN_LIFESPAN": ("lifespan", str),
        "UVICORN_ENV_FILE": ("env_file", str),
        "UVICORN_LOG_CONFIG": ("log_config", str),
        "UVICORN_SERVER_HEADER": ("server_header", _parse_bool),
        "UVICORN_DATE_HEADER": ("date_header", _parse_bool),
        "UVICORN_LIMIT_CONCURRENCY": (
            "limit_concurrency",
            lambda x: _parse_int(x, "UVICORN_LIMIT_CONCURRENCY"),
        ),
        "UVICORN_LIMIT_MAX_REQUESTS": (
            "limit_max_requests",
            lambda x: _parse_int(x, "UVICORN_LIMIT_MAX_REQUESTS"),
        ),
        "UVICORN_TIMEOUT_KEEP_ALIVE": (
            "timeout_keep_alive",
            lambda x: _parse_int(x, "UVICORN_TIMEOUT_KEEP_ALIVE"),
        ),
        "UVICORN_TIMEOUT_NOTIFY": (
            "timeout_notify",
            lambda x: _parse_int(x, "UVICORN_TIMEOUT_NOTIFY"),
        ),
        "UVICORN_SSL_KEYFILE": ("ssl_keyfile", str),
        "UVICORN_SSL_CERTFILE": ("ssl_certfile", str),
        "UVICORN_SSL_KEYFILE_PASSWORD": ("ssl_keyfile_password", str),
        "UVICORN_SSL_VERSION": (
            "ssl_version",
            lambda x: _parse_int(x, "UVICORN_SSL_VERSION"),
        ),
        "UVICORN_SSL_CERT_REQS": (
            "ssl_cert_reqs",
            lambda x: _parse_int(x, "UVICORN_SSL_CERT_REQS"),
        ),
        "UVICORN_SSL_CA_CERTS": ("ssl_ca_certs", str),
        "UVICORN_SSL_CIPHERS": ("ssl_ciphers", str),
        "UVICORN_HEADERS": ("headers", _parse_headers),
        "UVICORN_USE_COLORS": ("use_colors", _parse_bool),
        "UVICORN_UDS": ("uds", str),
        "UVICORN_FD": ("fd", lambda x: _parse_int(x, "UVICORN_FD")),
        "UVICORN_ROOT_PATH": ("root_path", str),
    }

    # Process advanced configuration options
    for env_var, (config_key, parser) in config_mapping.items():
        value = os.getenv(env_var)
        if value:
            try:
                config_kwargs[config_key] = parser(value)
            except ValueError as e:
                raise ValueError(f"Configuration error for {env_var}: {e}") from e

    return config_kwargs