import os

# print(os.path.dirname(__file__))
# Legacy JSON config example removed
from setup_master.master import Master
from utils.ai.ai_enums import Models

run = Master(working_dir=os.path.dirname(__file__))

run.AI.model = Models.GEMMA2_9B.model_id
a = run.AI.ollama_chat("this is a test")
