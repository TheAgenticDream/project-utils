# Project Utils

A comprehensive utility package providing database operations, AI integration, configuration management, and security functions for Python projects.

## Features

- **Database Operations**: PostgreSQL connection management, query execution, and ORM utilities via SQLAlchemy
- **AI Integration**: Unified interface for OpenAI and Ollama models with structured output support
- **Configuration Management**: Centralized configuration handling with environment variable support
- **Security**: Encryption/decryption utilities using Fernet symmetric encryption
- **Master Setup**: Automated project initialization and configuration

## Installation

### Using uv (Recommended)

```bash
# Add to pyproject.toml dependencies
dependencies = [
    "project-utils @ git+https://github.com/TheAgenticDream/project-utils.git@main",
]

# Then sync with uv
uv sync

# Or for a specific version/tag
dependencies = [
    "project-utils @ git+https://github.com/TheAgenticDream/project-utils.git@v0.1.0",
]
```

### Using pip (Alternative)

```bash
# Install from GitHub
pip install git+https://github.com/TheAgenticDream/project-utils.git

# Install specific version/tag
pip install git+https://github.com/TheAgenticDream/project-utils.git@v0.1.0

# In requirements.txt
git+https://github.com/TheAgenticDream/project-utils.git@main
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/TheAgenticDream/project-utils.git
cd project-utils

# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

## Configuration

The package expects certain environment variables to be set. Create a `.env` file in your project root:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database

# AI Configuration (Optional)
OPENAI_API_KEY=your_openai_key
OLLAMA_HOST=http://localhost:11434

# Security
ENCRYPTION_KEY=your_fernet_key
```

## Usage

### Database Operations

```python
from project_utils.utils.database.database_functions import DBFunctions
from project_utils.utils.config.config import Config

# Initialize configuration
config = Config()

# Create database functions instance
db = DBFunctions(
    db_host="localhost",
    db_port=5432,
    db_user="user",
    db_password="password",
    db_name="mydb"
)

# Execute a query
results = db.execute_query("SELECT * FROM users WHERE active = %s", (True,))

# Use with context manager for automatic connection handling
with db:
    users = db.fetch_all("SELECT * FROM users")
```

### AI Integration

```python
from project_utils.utils.ai.ai_driver import AIDriver
from project_utils.utils.ai.ai_enums import Models

# Initialize AI driver
ai = AIDriver()

# Generate text with OpenAI
response = ai.call_openai(
    prompt="Explain quantum computing",
    model=Models.GPT_4,
    temperature=0.7
)

# Generate structured output
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    sentiment: str
    key_points: list[str]

result = ai.call_openai_structured(
    prompt="Analyze this text: ...",
    model=Models.GPT_4,
    response_model=Analysis
)
```

### Configuration Management

```python
from project_utils.utils.config.config import Config

# Load configuration (automatically reads from .env)
config = Config()

# Access configuration values
db_host = config.get("DB_HOST", "localhost")
api_key = config.get("OPENAI_API_KEY")

# Set configuration values
config.set("CUSTOM_SETTING", "value")
```

### Security/Encryption

```python
from project_utils.utils.secrets.secrets import Encrypt

# Initialize encryption
encryptor = Encrypt()

# Encrypt data
encrypted = encryptor.encrypt("sensitive data")

# Decrypt data
decrypted = encryptor.decrypt(encrypted)

# Generate a new Fernet key
new_key = Encrypt.generate_key()
```

### Master Setup

```python
from project_utils.setup_master.master import Master

# Initialize and run setup
master = Master()
master.run_setup()  # Interactive setup process
```

## Project Structure

```
project-utils/
├── __init__.py
├── setup.py
├── pyproject.toml
├── README.md
├── setup_master/
│   ├── __init__.py
│   └── master.py
├── utils/
│   ├── __init__.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── ai_driver.py
│   │   ├── ai_basemodels.py
│   │   └── ai_enums.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── database_driver.py
│   │   └── database_functions.py
│   ├── exceptions/
│   │   ├── __init__.py
│   │   └── exceptions.py
│   └── secrets/
│       ├── __init__.py
│       └── secrets.py
└── build_utils/
    ├── __init__.py
    ├── autopep.py
    ├── build_docker.py
    └── file_collate.py
```

## Development

### Running Tests

```bash
# Using uv
uv run pytest
uv run pytest --cov=project_utils
uv run pytest tests/test_database.py

# Or using pytest directly (if in activated venv)
pytest
pytest --cov=project_utils
pytest tests/test_database.py
```

### Code Formatting

```bash
# Using uv
uv run black .
uv run ruff check .
uv run ruff format .
uv run mypy .

# Or with tools directly (if in activated venv)
black .
ruff check .
ruff format .
mypy .
```

## Requirements

- Python >= 3.10
- PostgreSQL (for database features)
- See `pyproject.toml` for full dependency list

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/TheAgenticDream/project-utils/issues) page.