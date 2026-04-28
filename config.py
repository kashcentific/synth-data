import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "openai/gpt-4o-mini"
TEMPERATURE = 0


# Metadata extraction config
SAMPLE_ROW_COUNT = 10

# Ratio below which object/text column is treated as categorical
CATEGORICAL_RATIO = 0.05