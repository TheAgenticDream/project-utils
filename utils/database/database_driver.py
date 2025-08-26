import time

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.exc import DisconnectionError, OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text

from ..config.config import Config


class DatabaseDriver:
    def __init__(
        self,
        user: str = None,
        password: str = None,
        host: str = None,
        port: str = None,
        database: str = None,
        maxcon: int = 10,
        mincon: int = 1,
    ):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.mincon = mincon
        self.maxcon = maxcon

        # Create SQLAlchemy engine with QueuePool
        self.engine = create_engine(
            self.get_connection_string(),
            pool_size=self.mincon,
            max_overflow=self.maxcon - self.mincon,
            poolclass=QueuePool,
            pool_pre_ping=True,  # Ensures stale connections are removed
        )

        self.session_factory = scoped_session(sessionmaker(bind=self.engine))

    def import_config(self):
        self.user = Config().return_config_database(database_user=True)
        self.password = Config().return_config_database(database_password=True)
        self.host = Config().return_config_database(database_host=True)
        self.port = Config().return_config_database(database_port=True)
        self.database = Config().return_config_database(database_database=True)
        self.maxcon = Config().return_config_database(database_max_connection=True)
        self.mincon = Config().return_config_database(database_min_connection=True)

    def get_connection_string(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def execute(
        self,
        execution_string: str,
        item_tuple: tuple = None,
        commit: bool = None,
        fetch_one: bool = None,
        fetch_all: bool = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Execute database query with retry logic for connection issues.

        Args:
            execution_string: SQL query string
            item_tuple: Query parameters
            commit: Whether to commit the transaction
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """

        for attempt in range(max_retries + 1):
            session = None
            try:
                session_maker = sessionmaker(bind=self.engine)
                session = session_maker()

                if item_tuple is not None:
                    result = session.execute(text(execution_string), item_tuple)
                else:
                    result = session.execute(text(execution_string))

                if commit:
                    session.commit()

                if fetch_one:
                    fetch = result.fetchone()
                elif fetch_all:
                    fetch = result.fetchall()
                else:
                    fetch = None

                session.close()
                return fetch

            except (DisconnectionError, OperationalError) as e:
                if session:
                    try:
                        session.rollback()
                        session.close()
                    except Exception:
                        pass

                if attempt < max_retries:
                    logger.warning(
                        f"Database connection error on attempt {attempt + 1}/{max_retries + 1}: {str(e)}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue
                else:
                    logger.error(
                        f"Failed to execute database query after {max_retries + 1} attempts: {str(e)}"
                    )
                    raise e

            except Exception as e:
                if session:
                    try:
                        session.rollback()
                        session.close()
                    except Exception:
                        pass
                logger.error(
                    f"Database query failed with non-retryable error: {str(e)}"
                )
                raise e


if __name__ == "__main__":
    pass
