import json
import os
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from ollama import Client as OllamaClient
from openai import OpenAI
from pydantic import BaseModel

from .ai_basemodels import convert_ollama_to_openai_usage
from .ai_enums import (
    ClientProvider,
    ClientProviderEndpoints,
    Models,
    get_japan_local_endpoint,
    get_ollama_local_endpoint,
)


class AIDriver:
    def __init__(
        self,
        open_router_key: str,
        open_ai_key: str,
        ollama_key: str,
        ai_model: str = None,
        agent_prompt: str = None,
        parser: BaseModel = None,
    ):
        """Initialize AI Driver with appropriate API keys and configurations."""
        logger.info("Initializing AIDriver with model: {}", ai_model or "None")
        self.open_ai_key = open_ai_key
        self.open_router_key = open_router_key
        self.ollama_key = ollama_key
        self.agent_prompt = agent_prompt or ""
        self.model = ai_model
        self.parser = parser
        self.client = None

        # Log configuration details without sensitive information
        logger.debug(
            "AIDriver configuration: model={}, parser={}, agent_prompt_length={}",
            self.model,
            self.parser.__name__ if self.parser else None,
            len(self.agent_prompt),
        )
        logger.success(
            "AIDriver initialized successfully with model: {}", self.model or "None"
        )

    def _truncate_text(self, text: str, max_length: int = 50) -> str:
        """Helper method to truncate text for logging purposes."""
        if not text or len(text) <= max_length:
            return text
        return f"{text[:max_length]}..."

    def _log_prompt(self, prompt: str, provider: str) -> None:
        """Standardized prompt logging with truncation."""
        truncated = self._truncate_text(prompt)
        logger.info(
            "Sending prompt to {} using model {}: {}", provider, self.model, truncated
        )
        logger.debug("Full prompt length: {}", len(prompt))

    def _log_response_received(self, response: Any, provider: str) -> None:
        """Log receipt of response with limited information."""
        # Log minimal information at info level
        logger.info("Received response from {} using model {}", provider, self.model)
        # Log more details at debug level
        if hasattr(response, "usage"):
            logger.debug(
                "Response usage stats for model {}: {}", self.model, response.usage
            )
        logger.trace(
            "Full response object from model {}: {}", self.model, response
        )  # Use trace for very verbose logs

    def _time_operation(self, operation: Callable, operation_name: str) -> Any:
        """Execute an operation with timing information."""
        start_time = time.time()
        try:
            result = operation()
            elapsed = time.time() - start_time
            logger.info(
                "{} with model {} completed in {:.2f}s",
                operation_name,
                self.model,
                elapsed,
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "{} with model {} failed after {:.2f}s: {}",
                operation_name,
                self.model,
                elapsed,
                str(e),
            )
            raise

    def open_ai_chat(self, prompt: str) -> tuple[str, dict] | str:
        """Send a chat request to OpenAI API."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt,
            client_provider=ClientProvider.OPEN_AI.value,
            system_prompt=self.agent_prompt,
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        def perform_openai_chat_operation():
            self._log_prompt(prompt, "OpenAI")

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            response_params = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt_str},
            ]

            system_not_supported = [Models.O1_MINI.model_id]
            if self.model in system_not_supported:
                logger.debug(
                    "Model {} doesn't support system messages, adapting prompt format",
                    self.model,
                )
                response_params = [
                    {"role": "user", "content": self.agent_prompt + prompt_str}
                ]

            model_id = self.model.split("/")[1]

            # Execute request with timing
            logger.debug("Executing chat completion with model: {}", model_id)
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=response_params,
                )
                elapsed = time.time() - start_time
                logger.info(
                    "OpenAI request with model {} completed in {:.2f}s",
                    model_id,
                    elapsed,
                )

                # Log response information
                self._log_response_received(response, "OpenAI")

                # Check response validity
                check_response_result = self.check_response(
                    response=response, client_provider=ClientProvider.OPEN_AI.value
                )
                if check_response_result is not True:
                    return check_response_result

                # Process response
                if response.choices:
                    content = response.choices[0].message.content
                    content_length = len(content) if content else 0
                    logger.success(
                        "Response from model {} retrieved successfully (length: {})",
                        model_id,
                        content_length,
                    )
                    return content, response.usage
                else:
                    logger.warning(
                        "No choices in OpenAI response for model {}", model_id
                    )
                    return "No response from the API."
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "OpenAI request with model {} failed after {:.2f}s: {}",
                    model_id,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_openai_chat_operation,
            client_provider=ClientProvider.OPEN_AI.value,
        )

    def gpt_parse(self, prompt: str) -> tuple[Any, dict] | str | bool:
        """Parse structured data using OpenAI's parsing capabilities."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.OPEN_AI.value
        )
        if check_initial_result is not True:
            return check_initial_result

        if not self.parser:
            logger.error(
                "Parser not configured for GPT parsing operation with model {}",
                self.model,
            )
            return False

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        def perform_gpt_parse_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info("Parsing prompt with GPT model {}: {}", self.model, truncated)
            logger.debug(
                "Using parser: {} with model {}", self.parser.__name__, self.model
            )

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            response_params = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt_str},
            ]

            system_not_supported = [Models.O1_MINI.model_id]
            if self.model in system_not_supported:
                logger.debug(
                    "Model {} doesn't support system messages, adapting prompt format",
                    self.model,
                )
                response_params = [
                    {"role": "user", "content": self.agent_prompt + prompt_str},
                ]

            model_id = self.model.split("/")[1]

            # Execute request with timing
            logger.debug("Executing parse operation with model: {}", model_id)
            try:
                start_time = time.time()
                response = self.client.beta.chat.completions.parse(
                    model=model_id,
                    messages=response_params,
                    response_format=self.parser,
                )
                elapsed = time.time() - start_time
                logger.info(
                    "GPT parse with model {} completed in {:.2f}s", model_id, elapsed
                )

                # Log response information
                logger.debug("Parse response received from model {}", model_id)
                logger.trace(
                    "Full parse response from model {}: {}", model_id, response
                )

                # Check response validity
                check_response_result = self.check_response(
                    response=response, client_provider=ClientProvider.OPEN_AI.value
                )
                if check_response_result is not True:
                    return check_response_result

                # Process response
                if response.choices:
                    parsed = response.choices[0].message.parsed
                    logger.success(
                        "Parsing with model {} successful using schema: {}",
                        model_id,
                        self.parser.__name__,
                    )
                    return parsed, response.usage
                else:
                    logger.warning(
                        "No parse choices in OpenAI response for model {}", model_id
                    )
                    return "No response from the API."
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "GPT parse with model {} failed after {:.2f}s: {}",
                    model_id,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_gpt_parse_operation,
            client_provider=ClientProvider.OPEN_AI.value,
        )

    """
        -----------------------------OPEN ROUTER
    """

    def open_router_chat(self, prompt: str) -> tuple[str, dict] | str:
        """Send a chat request to OpenRouter API."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.OPEN_ROUTER.value
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_ROUTER)

        def perform_chat_operation():
            # Log prompt information
            self._log_prompt(prompt, "OpenRouter")

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt_str},
                    ],
                    temperature=0.1,
                )
                elapsed = time.time() - start_time
                logger.info(
                    "OpenRouter request with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                self._log_response_received(response, "OpenRouter")

                # Check response validity
                check_response_result = self.check_response(
                    response=response, client_provider=ClientProvider.OPEN_ROUTER.value
                )
                if check_response_result is not True:
                    return check_response_result

                # Process response
                content = response.choices[0].message.content
                usage_stats = getattr(response, "usage", {})

                if not content:
                    logger.warning(
                        "Empty content in OpenRouter response for model {}", self.model
                    )
                    return "Empty content in response."

                content_length = len(content) if content else 0
                logger.success(
                    "OpenRouter response for model {} retrieved successfully (length: {})",
                    self.model,
                    content_length,
                )
                return content, usage_stats
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "OpenRouter request with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_chat_operation,
            client_provider=ClientProvider.OPEN_ROUTER.value,
        )

    def open_parser(self, prompt: str) -> tuple[Any, dict] | str:
        """Parse structured data using OpenRouter's parsing capabilities."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.OPEN_ROUTER.value
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_ROUTER)

        def perform_parse_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info(
                "Parsing prompt with OpenRouter model {}: {}", self.model, truncated
            )
            logger.debug(
                "Using parser: {} with model {}", self.parser.__name__, self.model
            )

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.beta.chat.completions.parse(
                    model=f"{self.model}",
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt_str},
                    ],
                    response_format=self.parser.model_json_schema(),
                )
                elapsed = time.time() - start_time
                logger.info(
                    "OpenRouter parse with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                logger.debug(
                    "Parse response received from OpenRouter model {}", self.model
                )
                logger.trace(
                    "Full parse response from model {}: {}", self.model, response
                )

                # Check response validity
                check_response_result = self.check_response(
                    response=response, client_provider=ClientProvider.OPEN_ROUTER.value
                )
                if check_response_result is not True:
                    return check_response_result

                # Process response
                parsed = response.choices[0].message.content
                try:
                    parsed = parsed.replace("```json", "")
                    parsed = parsed.replace("```", "")
                    parsed_content = json.loads(parsed)
                    result = self.parser(**parsed_content)
                    logger.success(
                        "OpenRouter parsing with model {} successful using schema: {}",
                        self.model,
                        self.parser.__name__,
                    )
                    return result, response.usage
                except Exception as parse_error:
                    logger.error(
                        "OpenRouter parsing with model {} failed, falling back to default model: {}",
                        self.model,
                        parse_error,
                    )
                    logger.debug(
                        "Failed parse content from model {}: {}",
                        self.model,
                        self._truncate_text(str(parsed), 100),
                    )
                    # Fallback to default model
                    original_model = self.model
                    self.model = Models.O3_MINI.model_id
                    logger.info(
                        "Falling back from model {} to {}", original_model, self.model
                    )
                    return self.gpt_parse(prompt=str(parsed))
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "OpenRouter parse with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_parse_operation,
            client_provider=ClientProvider.OPEN_ROUTER.value,
        )

    """
        -----------------------------OLLAMA
    """

    def ollama_chat(self, prompt: str) -> tuple[str, dict] | str:
        """Send a chat request to Ollama API."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.OLLAMA.value
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OLLAMA)

        def perform_ollama_chat_operation():
            # Log prompt information
            self._log_prompt(prompt, "Ollama")

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat(
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                )
                elapsed = time.time() - start_time
                logger.info(
                    "Ollama request with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                logger.info("Received response from Ollama model {}", self.model)
                logger.trace(
                    "Full Ollama response from model {}: {}", self.model, response
                )

                # Process response
                if not response:
                    logger.warning(
                        "No response from Ollama API for model {}", self.model
                    )
                    return "No response from the API."

                content = response.message.content
                usage = convert_ollama_to_openai_usage(
                    ollama_output=response.model_dump()
                )
                logger.debug("Ollama usage stats for model {}: {}", self.model, usage)

                content_length = len(content) if content else 0
                logger.success(
                    "Ollama response for model {} retrieved successfully (length: {})",
                    self.model,
                    content_length,
                )
                return content, usage
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "Ollama request with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_ollama_chat_operation,
            client_provider=ClientProvider.OLLAMA.value,
        )

    def ollama_chat_local(self, prompt: str) -> tuple[str, dict] | str:
        """Send a chat request to local Ollama API."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.OLLAMA_LOCAL.value
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OLLAMA_LOCAL)

        def perform_ollama_local_chat_operation():
            # Log prompt information
            self._log_prompt(prompt, "Local Ollama")

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat(
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                )
                elapsed = time.time() - start_time
                logger.info(
                    "Local Ollama request with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                logger.info("Received response from Local Ollama model {}", self.model)
                logger.trace(
                    "Full Local Ollama response from model {}: {}", self.model, response
                )

                # Process response
                if not response:
                    logger.warning(
                        "No response from Local Ollama API for model {}", self.model
                    )
                    return "No response from the API."

                content = response.message.content
                usage = convert_ollama_to_openai_usage(
                    ollama_output=response.model_dump()
                )
                logger.debug(
                    "Local Ollama usage stats for model {}: {}", self.model, usage
                )

                content_length = len(content) if content else 0
                logger.success(
                    "Local Ollama response for model {} retrieved successfully (length: {})",
                    self.model,
                    content_length,
                )
                return content, usage
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "Local Ollama request with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_ollama_local_chat_operation,
            client_provider=ClientProvider.OLLAMA_LOCAL.value,
        )

    def ollama_parser(self, prompt: str) -> tuple[Any, dict] | str:
        """Parse structured data using Ollama's capabilities."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.OLLAMA.value
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.OLLAMA)

        def perform_ollama_parser_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info(
                "Parsing prompt with Ollama model {}: {}", self.model, truncated
            )
            logger.debug(
                "Using parser: {} with model {}", self.parser.__name__, self.model
            )

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat(
                    messages=[
                        {"role": "system", "content": self.agent_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    format=self.parser.model_json_schema(),
                )
                elapsed = time.time() - start_time
                logger.info(
                    "Ollama parse with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                logger.debug("Parse response received from Ollama model {}", self.model)
                logger.trace(
                    "Full Ollama parse response from model {}: {}", self.model, response
                )

                # Process response
                if not response:
                    logger.warning(
                        "No parse response from Ollama API for model {}", self.model
                    )
                    return "No response from the API."

                parsed = response.message.content
                usage = convert_ollama_to_openai_usage(ollama_output=response.dict())
                logger.debug("Ollama usage stats for model {}: {}", self.model, usage)

                try:
                    parsed = parsed.replace("```json", "")
                    parsed = parsed.replace("```", "")
                    parsed_content = json.loads(parsed)
                    result = self.parser(**parsed_content)
                    logger.success(
                        "Ollama parsing with model {} successful using schema: {}",
                        self.model,
                        self.parser.__name__,
                    )
                    return result, usage
                except Exception as parse_error:
                    logger.error(
                        "Ollama parsing with model {} failed, falling back to default model: {}",
                        self.model,
                        parse_error,
                    )
                    logger.debug(
                        "Failed parse content from model {}: {}",
                        self.model,
                        self._truncate_text(str(parsed), 100),
                    )
                    # Fallback to default model
                    original_model = self.model
                    self.model = Models.O3_MINI.model_id
                    logger.info(
                        "Falling back from model {} to {}", original_model, self.model
                    )
                    return self.gpt_parse(prompt=str(parsed))
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "Ollama parse with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_ollama_parser_operation,
            client_provider=ClientProvider.OLLAMA.value,
        )

    """
        -----------------------------JAPAN LOCAL
    """

    def japan_local_chat(self, prompt: str) -> tuple[str, dict] | str:
        """Send a chat request to Japan DC local API (OpenAI-compatible)."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.JAPAN_LOCAL.value
        )
        if check_initial_result is not True:
            return check_initial_result

        # Set client
        self.set_client(client_provider=ClientProvider.JAPAN_LOCAL)

        def perform_japan_local_chat_operation():
            # Log prompt information
            self._log_prompt(prompt, "Japan Local")

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            response_params = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt_str},
            ]

            # Execute request with timing
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=response_params,
                    temperature=0.1,
                )
                elapsed = time.time() - start_time
                logger.info(
                    "Japan Local request with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                self._log_response_received(response, "Japan Local")

                # Check response validity
                check_response_result = self.check_response(
                    response=response, client_provider=ClientProvider.JAPAN_LOCAL.value
                )
                if check_response_result is not True:
                    return check_response_result

                # Process response
                content = response.choices[0].message.content
                usage_stats = getattr(response, "usage", {})

                if not content:
                    logger.warning(
                        "Empty content in Japan Local response for model {}", self.model
                    )
                    return "Empty content in response."

                content_length = len(content) if content else 0
                logger.success(
                    "Japan Local response for model {} retrieved successfully (length: {})",
                    self.model,
                    content_length,
                )
                return content, usage_stats
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "Japan Local request with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_japan_local_chat_operation,
            client_provider=ClientProvider.JAPAN_LOCAL.value,
        )

    def japan_local_parser(self, prompt: str) -> tuple[Any, dict] | str:
        """Parse structured data from Japan DC local API using vLLM guided JSON."""
        # Check initial conditions
        check_initial_result = self.check_initial(
            prompt=prompt, client_provider=ClientProvider.JAPAN_LOCAL.value
        )
        if check_initial_result is not True:
            return check_initial_result

        if not self.parser:
            logger.error(
                "Parser not configured for Japan Local parsing operation with model {}",
                self.model,
            )
            return False

        # Set client
        self.set_client(client_provider=ClientProvider.JAPAN_LOCAL)

        def perform_japan_local_parser_operation():
            # Log prompt information
            truncated = self._truncate_text(prompt)
            logger.info(
                "Parsing prompt with Japan Local model {}: {}", self.model, truncated
            )
            logger.debug(
                "Using parser: {} with model {}", self.parser.__name__, self.model
            )

            # Prepare request parameters
            self.agent_prompt = str(self.agent_prompt)
            prompt_str = str(prompt)

            response_params = [
                {"role": "system", "content": self.agent_prompt},
                {"role": "user", "content": prompt_str},
            ]

            # Generate Pydantic JSON schema for vLLM guided JSON
            try:
                json_schema = self.parser.model_json_schema()
                logger.debug(
                    "Generated JSON schema for vLLM guided JSON with model {}: {}",
                    self.model,
                    json.dumps(json_schema, indent=2)[:500] + "..."
                    if len(str(json_schema)) > 500
                    else json_schema,
                )
            except Exception as schema_error:
                logger.error(
                    "Failed to generate JSON schema for model {}: {}",
                    self.model,
                    str(schema_error),
                )
                return f"Schema generation failed: {str(schema_error)}"

            # Execute request with timing and guided JSON
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=response_params,
                    temperature=0.1,
                    extra_body={"guided_json": json_schema},
                )
                elapsed = time.time() - start_time
                logger.info(
                    "Japan Local structured parse with model {} completed in {:.2f}s",
                    self.model,
                    elapsed,
                )

                # Log response information
                logger.debug(
                    "Parse response received from Japan Local model {}", self.model
                )
                logger.trace(
                    "Full parse response from model {}: {}", self.model, response
                )

                # Check response validity
                check_response_result = self.check_response(
                    response=response, client_provider=ClientProvider.JAPAN_LOCAL.value
                )
                if check_response_result is not True:
                    return check_response_result

                # Process structured response from vLLM guided JSON
                content = response.choices[0].message.content
                usage_stats = getattr(response, "usage", {})

                if not content:
                    logger.warning(
                        "Empty content in Japan Local structured response for model {}",
                        self.model,
                    )
                    return "Empty content in response."

                # Parse structured JSON response directly to Pydantic model
                try:
                    parsed_json = json.loads(content)
                    result = self.parser(**parsed_json)

                    logger.success(
                        "Japan Local structured parsing with model {} successful using schema: {}",
                        self.model,
                        self.parser.__name__,
                    )
                    logger.debug(
                        "Structured response validated with attributes: {}",
                        list(result.__dict__.keys()),
                    )
                    return result, usage_stats

                except json.JSONDecodeError as json_error:
                    logger.error(
                        "JSON decode error in vLLM guided response from model {}: {}",
                        self.model,
                        str(json_error),
                    )
                    logger.debug(
                        "Invalid JSON content from model {}: {}",
                        self.model,
                        self._truncate_text(str(content), 200),
                    )
                    return f"JSON parsing failed: {str(json_error)}"

                except Exception as validation_error:
                    logger.error(
                        "Pydantic validation error for model {} with schema {}: {}",
                        self.model,
                        self.parser.__name__,
                        str(validation_error),
                    )
                    logger.debug(
                        "Validation failed for content from model {}: {}",
                        self.model,
                        self._truncate_text(str(content), 200),
                    )
                    return f"Model validation failed: {str(validation_error)}"
            except Exception as e:
                elapsed = time.time() - start_time if "start_time" in locals() else -1
                logger.error(
                    "Japan Local parse with model {} failed after {:.2f}s: {}",
                    self.model,
                    elapsed,
                    str(e),
                )
                raise

        return self.execute_with_error_handling(
            operation_func=perform_japan_local_parser_operation,
            client_provider=ClientProvider.JAPAN_LOCAL.value,
        )

    """
        -----------------------------NON LLM
    """

    def whisper(self, file_path: str) -> tuple[str, dict] | str:
        """
        Transcribe audio using OpenAI's Whisper model.

        Args:
            file_path (str): Path to the audio file to transcribe

        Returns:
            tuple[str, dict]: Tuple containing transcription text and usage stats
            str: Error message if transcription fails
        """
        logger.info("Transcribing audio file: {} with Whisper model", file_path)

        # Set client
        self.set_client(client_provider=ClientProvider.OPEN_AI)

        try:
            # Verify file exists and is accessible
            if not os.path.exists(file_path):
                logger.error("Audio file not found: {}", file_path)
                return "Audio file not found."

            if not os.access(file_path, os.R_OK):
                logger.error("Audio file not readable: {}", file_path)
                return "Audio file not readable."

            # Get file size for logging
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.debug("Audio file size: {:.2f} MB", file_size)

            # Execute request with timing
            start_time = time.time()
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            elapsed = time.time() - start_time
            logger.info(
                "Whisper transcription completed in {:.2f}s for {:.2f}MB file",
                elapsed,
                file_size,
            )

            # Log response information
            logger.debug("Transcription response received from Whisper model")
            logger.trace(
                "Full transcription response from Whisper model: {}", transcription
            )

            # Process response
            if hasattr(transcription, "text"):
                text_length = len(transcription.text) if transcription.text else 0
                logger.success(
                    "Whisper transcription completed successfully (length: {})",
                    text_length,
                )
                return transcription.text, getattr(transcription, "usage", {})
            else:
                logger.warning("No transcription text in Whisper response")
                return "No transcription text in response."

        except Exception as e:
            elapsed = time.time() - start_time if "start_time" in locals() else -1
            logger.error(
                "Whisper transcription failed after {:.2f}s: {}", elapsed, str(e)
            )
            return f"An error occurred during transcription: {str(e)}"

    def embeddings(self, prompt: str) -> tuple[Any, dict] | str:
        """Generate embeddings for the given prompt."""
        # Log prompt information
        truncated = self._truncate_text(prompt)
        logger.info("Generating embeddings with model {}: {}", self.model, truncated)

        # Set client
        self.set_client(ClientProvider.OPEN_AI)

        try:
            # Execute request with timing
            start_time = time.time()
            response = self.client.embeddings.create(model=self.model, input=prompt)
            elapsed = time.time() - start_time
            logger.info(
                "Embeddings generation with model {} completed in {:.2f}s",
                self.model,
                elapsed,
            )

            # Log response information
            logger.debug("Embeddings response received from model {}", self.model)
            logger.trace(
                "Full embeddings response from model {}: {}", self.model, response
            )

            # Process response
            if response.data:
                embedding = response.data[0].embedding
                embedding_size = len(embedding)
                logger.success(
                    "Embeddings with model {} generated successfully (dimensions: {})",
                    self.model,
                    embedding_size,
                )
                return embedding, response.usage
            else:
                logger.warning(
                    "No data in embeddings response from model {}", self.model
                )
                return "No response from the API."
        except Exception as e:
            elapsed = time.time() - start_time if "start_time" in locals() else -1
            logger.error(
                "Embeddings generation with model {} failed after {:.2f}s: {}",
                self.model,
                elapsed,
                str(e),
            )
            return f"An error occurred: {str(e)}"

    """
    -----------------------------UTILS
    """

    def set_client(self, client_provider: ClientProvider) -> None:
        """Set the appropriate client based on the provider, honoring env overrides for base URLs."""
        openai_base_url = os.getenv(
            "OPENAI_BASE_URL", ClientProviderEndpoints.OPEN_AI.value
        )
        openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", ClientProviderEndpoints.Open_Router.value
        )
        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL", ClientProviderEndpoints.OLLAMA.value
        )
        ollama_local_base_url = get_ollama_local_endpoint()
        japan_local_base_url = get_japan_local_endpoint()

        client_mapper = {
            ClientProvider.OPEN_AI: lambda: OpenAI(
                api_key=self.open_ai_key, base_url=openai_base_url
            ),
            ClientProvider.OPEN_ROUTER: lambda: OpenAI(
                api_key=self.open_router_key, base_url=openrouter_base_url
            ),
            ClientProvider.OLLAMA: lambda: OllamaClient(
                headers=(
                    {"Authorization": f"Bearer {self.ollama_key}"}
                    if self.ollama_key
                    else None
                ),
                host=ollama_base_url,
            ),
            ClientProvider.OLLAMA_LOCAL: lambda: OllamaClient(
                host=ollama_local_base_url
            ),
            ClientProvider.JAPAN_LOCAL: lambda: OpenAI(
                api_key="dummy", base_url=japan_local_base_url
            ),
        }

        logger.info("Setting client to {} for model {}", client_provider, self.model)

        client_factory = client_mapper.get(client_provider)
        if client_factory is None:
            error_msg = (
                f"Unsupported client provider: {client_provider} for model {self.model}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.client = client_factory()
        logger.success(
            "Client successfully set to {} for model {}", client_provider, self.model
        )

    def check_initial(
        self,
        prompt: Any,
        client_provider: ClientProvider,
        system_prompt: str | None = None,
    ) -> bool | str:
        """Check initial conditions before sending request."""
        # ---------------- type checks ----------------
        if not isinstance(prompt, str):
            logger.error(
                "Invalid prompt type: {} for model {}. Expected str.",
                type(prompt).__name__,
                self.model,
            )
            return "Invalid prompt type. Expected string."

        if system_prompt is not None and not isinstance(system_prompt, str):
            logger.error(
                "Invalid system_prompt type: {} for model {}. Expected str or None.",
                type(system_prompt).__name__,
                self.model,
            )
            return "Invalid system_prompt type. Expected string."

        # ---------------- content checks ----------------
        prompt_is_empty = not prompt.strip()
        system_is_empty = system_prompt is None or not system_prompt.strip()

        if prompt_is_empty and system_is_empty:
            logger.warning(
                "Both prompt and system_prompt are empty for model {}", self.model
            )
            return "Empty prompt returned."

        # ---------------- model check ----------------
        if not self.model:
            logger.error("No model specified for {} request", client_provider)
            return "No model specified."

        logger.debug(
            "Initial checks passed for {} request with model {}",
            client_provider,
            self.model,
        )
        return True

    def check_response(
        self, response: Any, client_provider: ClientProvider
    ) -> bool | str:
        """Check response validity."""
        # Check response existence
        if not response:
            logger.error(
                "Received empty response from {} for model {}",
                client_provider,
                self.model,
            )
            return "Empty response received from API."

        # Check response format
        if not hasattr(response, "choices"):
            logger.error(
                "Invalid response format from {} for model {}: missing 'choices' attribute",
                client_provider,
                self.model,
            )
            return "Invalid response format from API."

        # Check choices content
        if not response.choices:
            logger.warning(
                "No choices in response from {} for model {}",
                client_provider,
                self.model,
            )
            return "No response choices available."

        logger.debug(
            "Response checks passed for {} response with model {}",
            client_provider,
            self.model,
        )
        return True

    def execute_with_error_handling(
        self, operation_func: Callable, client_provider: str
    ) -> Any:
        """
        Execute an operation with standardized error handling.

        Args:
            operation_func: Callable function to execute within error handling
            client_provider: The client provider name for error messages

        Returns:
            The result of operation_func or an error message
        """
        try:
            return operation_func()
        except ConnectionError as e:
            logger.error(
                "Connection error while calling {} with model {}: {}",
                client_provider,
                self.model,
                str(e),
            )
            return f"Connection error: {str(e)}"
        except TimeoutError as e:
            logger.error(
                "Timeout while calling {} with model {}: {}",
                client_provider,
                self.model,
                str(e),
            )
            return f"Request timed out: {str(e)}"
        except Exception as e:
            logger.error(
                "Error during {} operation with model {}: {}",
                client_provider,
                self.model,
                str(e),
            )
            logger.exception(
                f"Full exception details for model {self.model}"
            )  # This logs the stack trace
            return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    run = AIDriver
