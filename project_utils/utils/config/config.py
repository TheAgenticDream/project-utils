import base64
import json
import os
from pathlib import Path
from typing import Any


class Config:
    """
    Configuration manager with environment-only approach.
    Single source of truth: .env file (no JSON fallback needed)
    """

    def __init__(self, config_file: str | None = None, kubernetes: bool | None = None):
        """
        Initialize config with environment-only approach.

        Args:
            config_file: Optional path to specific config file (legacy support)
            kubernetes: Legacy parameter for backward compatibility (ignored)
        """
        self.data = {}

        # Load .env file - this is our single source of truth
        self._load_env_file()

        # Legacy support: if a specific config file is provided, load it
        if config_file:
            self._load_config_file(config_file)

    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        try:
            project_root = self._find_project_root()
            env_file = project_root / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            # Only set if not already in environment
                            if key not in os.environ:
                                os.environ[key] = value
        except Exception:
            # Silently continue if .env loading fails
            pass

    def _find_project_root(self) -> Path:
        """Find the project root directory by looking for justfile."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "justfile").exists():
                return parent
        raise FileNotFoundError("Could not find project root (no justfile found)")

    def _load_config_file(self, config_file: str):
        """Load secrets from JSON config file (structured encoded format).
        
        Non-secret configuration continues to come from .env files.
        This method only processes secrets from the JSON config.
        """
        try:
            with open(config_file) as f:
                config_data = json.load(f)
                
            # Load secrets from structured JSON config into environment variables
            self._load_secrets_from_json_to_env(config_data)
        except FileNotFoundError:
            # Config file not found - this is expected in containerized environments
            # where config is mounted at runtime or provided via environment variables
            print(f"Info: Config file {config_file} not found, using environment variables only")
        except Exception as e:
            # Other errors should still be logged
            print(f"Warning: Could not load config file {config_file}: {e}")

    def _load_secrets_from_json_to_env(self, config_data: dict):
        """Decode structured JSON config and set secrets as environment variables."""
        try:
            # Process each section that has encoded secrets
            for section_name, section_data in config_data.items():
                if isinstance(section_data, dict) and section_data.get("encoded", False):
                    # Decode each field in this section
                    for key, encoded_value in section_data.items():
                        if key == "encoded":
                            continue
                        
                        try:
                            # Decode base64 value
                            decoded_value = base64.b64decode(encoded_value).decode().strip()
                            
                            # Map to environment variable names
                            env_var_name = self._map_json_key_to_env_var(section_name, key)
                            if env_var_name:
                                os.environ[env_var_name] = decoded_value
                                
                        except Exception as e:
                            print(f"Warning: Could not decode {section_name}.{key}: {e}")
                            
        except Exception as e:
            raise ValueError(f"Failed to load secrets from JSON config: {e}")

    def _map_json_key_to_env_var(self, section: str, key: str) -> str:
        """Map JSON config keys to environment variable names."""
        mapping = {
            # Secrets section
            ("secrets", "fernet_key"): "FERNET_KEY",
            ("secrets", "salt_key"): "SALT_KEY", 
            ("secrets", "session_secret"): "WEBSITE_SESSION_SECRET",
            
            # API Keys section
            ("api_keys", "openai"): "OPENAI_API_KEY",
            ("api_keys", "openrouter"): "OPENROUTER_API_KEY",
            ("api_keys", "orchestration"): "ORCHESTRATION_INTEGRATION_API_KEY",
            
            # Database credentials section
            ("database_credentials", "password"): "DB_PASSWORD",
        }
        
        return mapping.get((section, key))

    # Database configuration - environment only
    @property
    def database_host(self) -> str:
        return os.getenv("DB_HOST", "127.0.0.1")

    @property
    def database_port(self) -> int:
        return int(os.getenv("DB_PORT", "5432"))

    @property
    def database_name(self) -> str:
        # Check for service-specific database names first
        orchestration_db = os.getenv("ORCHESTRATION_DB_NAME")
        website_db = os.getenv("WEBSITE_DB_NAME")

        # Try to determine which service is calling this by checking the current working directory
        current_dir = os.getcwd()
        if "WebsiteAPI" in current_dir:
            return website_db or "website_api_db"
        elif "Orchestration" in current_dir:
            return orchestration_db or "orchestration"
        else:
            # For projects-utils or other contexts, default to orchestration database
            return orchestration_db or os.getenv("DB_NAME", "orchestration")

    @property
    def database_user(self) -> str:
        return os.getenv("DB_USER", "project_friday_user")

    @property
    def database_password(self) -> str:
        return os.getenv("DB_PASSWORD", "")

    @property
    def database_max_connections(self) -> int:
        return int(os.getenv("DB_MAX_CONNECTIONS", "20"))

    @property
    def database_min_connections(self) -> int:
        return int(os.getenv("DB_MIN_CONNECTIONS", "1"))

    @property
    def test_work_package_uuid(self) -> str:
        """Get test work package UUID from environment"""
        return os.getenv("TEST_WORK_PACKAGE_UUID", "")

    # Secrets configuration - environment only
    @property
    def fernet_key(self) -> str:
        return os.getenv("FERNET_KEY", "")

    @property
    def salt_key(self) -> str:
        return os.getenv("SALT_KEY", "")

    # AI Keys configuration - environment only
    @property
    def openai_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def openrouter_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def ollama_key(self) -> str:
        return os.getenv("OLLAMA_KEY", "")

    # Orchestration configuration - environment only
    @property
    def orchestration_host(self) -> str:
        return os.getenv("ORCHESTRATION_HOST", "localhost")

    @property
    def orchestration_port(self) -> str:
        return os.getenv("ORCHESTRATION_PORT", "5000")

    @property
    def orchestration_api_key(self) -> str:
        return os.getenv("ORCHESTRATION_API_KEY", "")

    @property
    def orchestration_base_url(self) -> str:
        """Base URL for calling the Orchestration service from internal clients."""
        explicit = os.getenv("ORCHESTRATION_BASE_URL")
        if explicit:
            return explicit
        host = self.orchestration_host
        port = self.orchestration_port
        scheme = os.getenv("ORCHESTRATION_SCHEME", "http")
        return f"{scheme}://{host}:{port}"

    @property
    def orchestration_integration_api_key(self) -> str:
        """API key used by internal services to talk to Orchestration."""
        return os.getenv("ORCHESTRATION_INTEGRATION_API_KEY", "")

    # Website API configuration - environment only
    @property
    def website_host(self) -> str:
        return os.getenv("WEBSITE_HOST", "localhost")

    @property
    def website_port(self) -> str:
        return os.getenv("WEBSITE_PORT", "5001")

    @property
    def website_session_secret(self) -> str:
        return os.getenv("WEBSITE_SESSION_SECRET", "")

    @property
    def website_cors_origins(self) -> list[str]:
        raw = os.getenv("WEBSITE_CORS_ORIGINS")
        if not raw:
            # Sensible defaults for local dev using configurable frontend port
            frontend_port = os.getenv("FRONTEND_PORT", "3000")
            return [
                f"http://localhost:{frontend_port}",
                f"http://localhost:{int(frontend_port) + 1}",
                f"http://localhost:{int(frontend_port) + 2}",
                f"http://localhost:{int(frontend_port) + 3}",
            ]
        # Support comma and whitespace separated lists
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        return parts

    # Legacy compatibility methods - maintain the old interface for backward compatibility
    def return_config_database(self, **kwargs) -> Any:
        """Legacy method for database config access."""
        if kwargs.get("database_encoded"):
            return False  # We don't encode anymore with environment variables
        if kwargs.get("database_host"):
            return self.database_host
        if kwargs.get("database_port"):
            return self.database_port
        if kwargs.get("database_database"):
            return self.database_name
        if kwargs.get("database_user"):
            return self.database_user
        if kwargs.get("database_password"):
            return self.database_password
        if kwargs.get("database_max_connection"):
            return self.database_max_connections
        if kwargs.get("database_min_connection"):
            return self.database_min_connections
        return {}  # Return empty dict for compatibility

    def return_config_secrets(self, **kwargs) -> Any:
        """Legacy method for secrets config access."""
        if kwargs.get("secrets_encoded"):
            return False  # We don't encode anymore with environment variables
        if kwargs.get("secrets_fernet_key"):
            return self.fernet_key
        if kwargs.get("secrets_salt_key"):
            return self.salt_key
        return {}  # Return empty dict for compatibility

    def return_config_ai_keys(self, **kwargs) -> Any:
        """Legacy method for AI keys config access."""
        if kwargs.get("ai_keys_encoded"):
            return False  # We don't encode anymore with environment variables
        if kwargs.get("ai_keys_OpenAi_key"):
            return self.openai_key
        if kwargs.get("ai_keys_OpenRouter_Key"):
            return self.openrouter_key
        if kwargs.get("ai_keys_Ollama_Key"):
            return self.ollama_key
        return {}  # Return empty dict for compatibility

    def return_config_orchestration_engine(self, **kwargs) -> Any:
        """Legacy method for orchestration config access."""
        if kwargs.get("orchestration_engine_host"):
            return self.orchestration_host
        if kwargs.get("orchestration_engine_port"):
            return self.orchestration_port
        if kwargs.get("orchestration_engine_api_key"):
            return self.orchestration_api_key
        return {}  # Return empty dict for compatibility

    def return_config_orchestration_integration(self, **kwargs) -> Any:
        """Legacy method for orchestration integration config access."""
        if kwargs.get("orchestration_integration_base_url"):
            return self.orchestration_base_url
        if kwargs.get("orchestration_integration_api_key"):
            return self.orchestration_integration_api_key
        return {}  # Return empty dict for compatibility

    def return_config_website_api(self, **kwargs) -> Any:
        """Legacy method for website API config access."""
        if kwargs.get("website_api_host"):
            return self.website_host
        if kwargs.get("website_api_port"):
            return self.website_port
        if kwargs.get("website_session_secret"):
            return self.website_session_secret
        if kwargs.get("website_cors_origins"):
            return self.website_cors_origins
        return {}


if __name__ == "__main__":
    # Test the configuration
    try:
        config = Config()
    except Exception:
        pass
