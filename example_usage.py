#!/usr/bin/env python3
"""
Example usage of the project-utils package.
This file demonstrates how to use the package after installation.
"""

import os
from pathlib import Path

# After package installation, you would import like this:
# from project_utils import Master, AIDriver, DBFunctions, Encrypt

# For local development, use:
try:
    from project_utils import Master, AIDriver, DBFunctions, Encrypt
except ImportError:
    # Fallback for local development
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from setup_master.master import Master
    from utils.ai.ai_driver import AIDriver
    from utils.database.database_functions import DBFunctions
    from utils.secrets.secrets import Encrypt


def example_encryption():
    """Demonstrate encryption/decryption."""
    print("\n=== Encryption Example ===")
    
    # Initialize encryptor (will use ENCRYPTION_KEY from environment)
    encryptor = Encrypt()
    
    # Encrypt some data
    secret_data = "This is sensitive information"
    encrypted = encryptor.encrypt(secret_data)
    print(f"Original: {secret_data}")
    print(f"Encrypted: {encrypted[:50]}...")  # Show first 50 chars
    
    # Decrypt it back
    decrypted = encryptor.decrypt(encrypted)
    print(f"Decrypted: {decrypted}")


def example_database():
    """Demonstrate database operations."""
    print("\n=== Database Example ===")
    
    # Get database config from environment
    db_config = {
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_port': int(os.getenv('DB_PORT', 5432)),
        'db_user': os.getenv('DB_USER', 'test_user'),
        'db_password': os.getenv('DB_PASSWORD', 'test_pass'),
        'db_name': os.getenv('DB_NAME', 'test_db')
    }
    
    try:
        # Initialize database functions
        db = DBFunctions(**db_config)
        print(f"Database connection configured for {db_config['db_host']}:{db_config['db_port']}")
        
        # Example query (would work if database is available)
        # results = db.execute_query("SELECT version()")
        # print(f"Database version: {results}")
    except Exception as e:
        print(f"Database connection example (not connected): {e}")


def example_ai():
    """Demonstrate AI integration."""
    print("\n=== AI Integration Example ===")
    
    try:
        # Initialize AI driver
        ai = AIDriver()
        
        # Check if OpenAI is configured
        if os.getenv('OPENAI_API_KEY'):
            print("OpenAI API key detected - ready for OpenAI calls")
            # Example call (commented to avoid API costs):
            # response = ai.call_openai(
            #     prompt="Hello, how are you?",
            #     model="gpt-3.5-turbo"
            # )
            # print(f"AI Response: {response}")
        else:
            print("No OpenAI API key found - set OPENAI_API_KEY environment variable")
            
        # Check if Ollama is available
        if os.getenv('OLLAMA_HOST'):
            print(f"Ollama host configured: {os.getenv('OLLAMA_HOST')}")
        else:
            print("No Ollama host configured - set OLLAMA_HOST environment variable")
            
    except Exception as e:
        print(f"AI initialization example: {e}")


def example_master_setup():
    """Demonstrate Master setup utility."""
    print("\n=== Master Setup Example ===")
    
    try:
        # Initialize Master for project setup
        master = Master()
        print("Master setup initialized")
        print("Run master.run_setup() for interactive project configuration")
        
        # Get current configuration
        if hasattr(master, 'config'):
            print(f"Configuration loaded from: {master.config.config_file}")
    except Exception as e:
        print(f"Master setup example: {e}")


def main():
    """Run all examples."""
    print("=" * 50)
    print("Project Utils - Usage Examples")
    print("=" * 50)
    
    # Load environment variables from .env if it exists
    from pathlib import Path
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        print(f"Loading configuration from {env_file}")
        # In production, you'd use python-dotenv:
        # from dotenv import load_dotenv
        # load_dotenv()
    else:
        print("No .env file found - using system environment variables")
        print("Copy .env.example to .env and configure for your environment")
    
    # Run examples
    example_encryption()
    example_database()
    example_ai()
    example_master_setup()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()