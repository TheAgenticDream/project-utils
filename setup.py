from setuptools import find_packages, setup

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="project-utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "setuptools",
        "autopep8",
        "psycopg2>=2.9.10",
        "sqlalchemy>=2.0.38",
        "cryptography>=3.4.0",
        "passlib>=1.7.4",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "loguru>=0.6.0",
        "ollama",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest-cov>=4.0.0",
        ]
    },
    python_requires=">=3.10",
    description="Shared utilities for database, AI, configuration and security operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheAgenticDream/project-utils",
    author="Thomas Tiotto",
    author_email="thomas.tiotto@kyndryl.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        "project_utils.utils.config": ["*.json"],
    },
    include_package_data=True,
)
