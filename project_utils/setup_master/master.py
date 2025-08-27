import os

from loguru import logger

from ..utils.ai.ai_driver import AIDriver as AI
from ..utils.config.config import Config as CF
from ..utils.database.database_driver import DatabaseDriver as DB
from ..utils.exceptions.exceptions import DatabaseError, GPTError, SecurityError
from ..utils.secrets.secrets import Encrypt as SE


class Master:
    def __init__(
        self,
        working_dir: str = None,
        config_file: str = None,
        default_model: str = None,
    ):
        self.working_directory = working_dir
        self.config_file = config_file
        self.default_model = default_model
        self.set_working_directory()
        logger.info("STARTING MASTER")

        self.CF: CF | None = None
        self.SE: SE | None = None
        self.DB: DB | None = None
        self.AI: AI | None = None

        self.load_configuration()
        self.temp_dir = "TEMP_STORE"

        self.set_security()
        self.set_database_manager()
        self.set_ai()

        self.global_process_limit = 25

    def load_configuration(self):
        """Load configuration from environment and optional JSON config file.
        
        Uses environment variables as primary source, with optional JSON config
        file for base64-encoded secrets.
        """
        if self.config_file:
            logger.info(f"Loading configuration with JSON config file: {self.config_file}")
        else:
            logger.info("Loading configuration from environment only")
            
        self.CF = CF(config_file=self.config_file, kubernetes=False)

    def set_security(self):
        try:
            logger.info("Setting up security...")
            key_location = os.getenv("KEY_LOCATION", "")

            # Get secret values directly from environment-config
            fernet_key = self.CF.return_config_secrets(secrets_fernet_key=True)
            salt_key = self.CF.return_config_secrets(secrets_salt_key=True)

            self.SE = SE(
                key_location=key_location, secret_key=fernet_key, salt_key=salt_key
            )
            logger.success("SECRETS initialized")
        except Exception as e:
            logger.error(f"SECRETS error: {e}")
            raise SecurityError(f"Failed to initialize secrets: {e}") from e

    def set_database_manager(self):
        try:
            logger.info("Setting up the database manager...")

            # Get database values
            user = self.CF.return_config_database(database_user=True)
            password = self.CF.return_config_database(database_password=True)
            host = self.CF.return_config_database(database_host=True)
            port = self.CF.return_config_database(database_port=True)
            database = self.CF.return_config_database(database_database=True)

            self.DB = DB(
                user=user,
                password=password,
                host=host,
                port=port,
                database=database,
                maxcon=self.CF.return_config_database(database_max_connection=True),
                mincon=self.CF.return_config_database(database_min_connection=True),
            )
            logger.success("Database manager setup successfully.")
        except Exception as e:
            logger.error(f"Failed to set up the database manager: {e}")
            raise DatabaseError(f"Failed to set up the database manager: {e}") from e

    def set_ai(self):
        try:
            logger.info("Setting up the GPT class...")

            # Get AI keys
            openrouter_key = self.CF.return_config_ai_keys(ai_keys_OpenRouter_Key=True)
            ollama_key = self.CF.return_config_ai_keys(ai_keys_Ollama_Key=True)
            openai_key = self.CF.return_config_ai_keys(ai_keys_OpenAi_key=True)

            # Use default model if provided, otherwise None (will be set later when needed)
            ai_model = self.default_model

            self.AI = AI(
                open_router_key=openrouter_key,
                ollama_key=ollama_key,
                open_ai_key=openai_key,
                ai_model=ai_model,
            )

            if ai_model:
                logger.success("GPT initialized with model: {}", ai_model)
            else:
                logger.success(
                    "GPT initialized without default model (will be set per request)"
                )
        except Exception as e:
            logger.error(f"GPT initialization error: {e}")
            raise GPTError(f"Failed to initialize GPT: {e}") from e

    def set_working_directory(self):
        try:
            logger.info("Setting working directory")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if self.working_directory is None:
                os.chdir(script_dir)
                logger.success(f"Working directory set to {script_dir}")
            else:
                os.chdir(self.working_directory)
                logger.success(f"Working directory set to {self.working_directory}")
        except Exception as e:
            logger.error(f"Failed to set working directory: {e}")
            raise

    def set_ai_model(self, model: str):
        """Set the AI model for the current session."""
        if self.AI:
            self.AI.model = model
            logger.info("AI model set to: {}", model)
        else:
            logger.warning("AI component not initialized yet")

    def get_available_models(self):
        """Return a list of available models for reference."""
        from utils.ai.ai_enums import Models

        return {
            "openai": [
                model.model_id for model in Models if model.provider == "open_ai"
            ],
            "openrouter": [
                model.model_id for model in Models if model.provider == "open_router"
            ],
            "ollama": [
                model.model_id for model in Models if model.provider == "ollama"
            ],
            "japan_local": [
                model.model_id for model in Models if model.provider == "japan_local"
            ],
        }


if __name__ == "__main__":
    test_run = Master(working_dir=os.getcwd())
