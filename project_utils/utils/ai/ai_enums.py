import os
from enum import Enum


class Models(Enum):
    DEEPSEEK_R1 = ("deepseek/deepseek-r1", "open_router")
    DEEPSEEK = ("deepseek/deepseek-chat", "open_router")
    SONNET_3_5 = ("anthropic/claude-3.5-sonnet:beta", "open_router")
    SONNET_3_7 = ("anthropic/claude-3.7-sonnet:beta", "open_router")
    SONNET_THINKING = ("anthropic/claude-3.7-sonnet:thinking", "open_router")
    MISTRAL_SMALL = ("mistralai/mistral-small-3.1-24b-instruct", "open_router")
    COMMAND_A = ("cohere/command-a", "open_router")
    COMMAND_A_7B = ("cohere/command-r7b-12-2024", "open_router")
    QWEN_32B = ("qwen/qwq-32b", "open_router")
    MISTRAL_SMALL_INSTRUCT = (
        "mistralai/mistral-small-24b-instruct-2501",
        "open_router",
    )
    LLAMA_3_3_70B = ("meta-llama/llama-3.3-70b-instruct", "open_router")
    GEMINI_FLASH_2 = ("google/gemini-2.0-flash-001", "open_router")
    MINMAX = ("minimax/minimax-01", "open_router")
    PHI4 = ("microsoft/phi-4", "open_router")
    O3_MINI_HIGH = ("openai/o3-mini-high", "open_router")

    # Japan DC Local Models
    QWEN3_30B_JAPAN_DC = ("qwen3-30b-a3b-awq", "japan_local")
    GPT_OSS_120B_JAPAN_DC = ("gpt-oss-120b", "japan_local")

    O1 = ("openai/o1-2024-12-17", "open_ai")
    O1_MINI = ("openai/o1-mini", "open_ai")
    O3_MINI = ("openai/o3-mini", "open_ai")
    GPT_4O = ("openai/gpt-4o-2024-08-06", "open_ai")
    GPT_4O_MINI = ("openai/gpt-4o-mini", "open_ai")

    GEMMA3_27B = ("gemma3:27b", "ollama")
    GEMMA3_4B = ("gemma3:4b", "ollama")
    QWQ_LATEST = ("qwq:latest", "ollama")
    MISTRAL_SMALL_24B = ("mistral-small:24b", "ollama")
    DEEPSEEK_R1_32B = ("deepseek-r1:32b", "ollama")
    DEEPSEEK_R1_14B = ("deepseek-r1:14b", "ollama")
    PHI4_LATEST = ("phi4:latest", "ollama")
    DOLPHIN_LLAMA3_LATEST = ("dolphin-llama3:latest", "ollama")
    TULU3_LATEST = ("tulu3:latest", "ollama")
    MARCO_O1_LATEST = ("marco-o1:latest", "ollama")
    QWEN2_5_LATEST = ("qwen2.5:latest", "ollama")
    PHI3_3_8B = ("phi3:3.8b", "ollama")
    MISTRAL_7B = ("mistral:7b", "ollama")
    MISTRAL_NEMO_LATEST = ("mistral-nemo:latest", "ollama")
    LLAVA_7B = ("llava:7b", "ollama")
    LLAVA_13B = ("llava:13b", "ollama")
    LLAMA3_1_8B = ("llama3.1:8b", "ollama")
    CODESTRAL_LATEST = ("codestral:latest", "ollama")
    LLAMA3_2_VISION_LATEST = ("llama3.2-vision:latest", "ollama")
    MINICPM_V_LATEST = ("minicpm-v:latest", "ollama")

    def __init__(self, model_id, provider):
        self._model_id = model_id
        self.provider = provider

    @property
    def model_id(self):
        return self._model_id

    @classmethod
    def model_ids(cls):
        """Return a list of all model IDs"""
        return [model.model_id for model in cls]

    @classmethod
    def model_names(cls):
        """Return a list of all enum member names --"""
        return [model.name for model in cls]

    @classmethod
    def get_provider_by_name(cls, name):
        """Return the provider for a given model name"""
        try:
            return cls[name].provider
        except KeyError:
            return None

    @classmethod
    def get_model_id_by_name(cls, name):
        """Return the model ID for a given model name"""
        try:
            return cls[name].model_id
        except KeyError:
            return None


class ClientProvider(Enum):
    OPEN_AI = "open_ai"
    OPEN_ROUTER = "open_router"
    OLLAMA = "ollama"
    OLLAMA_LOCAL = "ollama_local"
    JAPAN_LOCAL = "japan_local"


class ClientProviderEndpoints(Enum):
    OPEN_AI = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    Open_Router = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OLLAMA = os.getenv("OLLAMA_BASE_URL", "https://chat.kxsb.org/ollama")
    # OLLAMA_LOCAL removed - use get_ollama_local_endpoint() function instead


def get_ollama_local_endpoint():
    """Get Ollama local endpoint from environment or use default"""
    return os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434")


def get_japan_local_endpoint():
    """Get Japan DC local endpoint from environment or use default"""
    return os.getenv("JAPAN_LOCAL_BASE_URL", "http://localhost:8000/v1")
