import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI / OpenRouter API Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "openai/gpt-4o-mini"
TEMPERATURE = 0

# Metadata extraction: number of sample rows sent to Thinker prompt
SAMPLE_ROW_COUNT = 50

# Ratio below which an object/text column is treated as categorical
CATEGORICAL_RATIO = 0.05

# Max rows the evaluator will process (caps huge datasets)
MAX_ROWS_FOR_EVAL = 500

# Output report path (relative to project root)
REPORT_OUTPUT_PATH = "audit_report.json"