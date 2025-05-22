import os
import logging
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv

# Initialize a logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers: # Basic config if no global logger is set for this module
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Find .env file automatically by searching up the directory tree from the current file's location or CWD.
# This makes it robust for different execution contexts (e.g., running scripts from root or from src/utils).
dotenv_path = find_dotenv(usecwd=True) # Prefer .env in current working directory if running scripts from root

if dotenv_path:
    logger.info(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning(".env file not found. Using default configurations or environment variables if set elsewhere.")

def get_env_var(variable_name: str, default_value: Optional[str] = None) -> Optional[str]:
    """
    Retrieves an environment variable.

    Args:
        variable_name: The name of the environment variable.
        default_value: The default value to return if the variable is not found.

    Returns:
        The value of the environment variable or the default value.
    """
    value = os.getenv(variable_name, default_value)
    if value is None and default_value is None:
        logger.warning(f"Environment variable '{variable_name}' not found and no default value provided.")
    elif value == default_value:
        logger.debug(f"Environment variable '{variable_name}' not found; using default value: '{default_value}'.")
    else:
        logger.debug(f"Environment variable '{variable_name}' found with value: '{value[:20]}...' (value truncated for logs if long).")
    return value

def get_list_env_var(variable_name: str, default_value: str = "") -> List[str]:
    """
    Retrieves an environment variable that is a comma-separated list.

    Args:
        variable_name: The name of the environment variable.
        default_value: The default comma-separated string if the variable is not found.

    Returns:
        A list of strings. Returns an empty list if the variable is empty or not found and default is empty.
    """
    value_str = get_env_var(variable_name, default_value)
    if not value_str: # Handles None or empty string
        return []
    return [item.strip() for item in value_str.split(',') if item.strip()]

# --- Financial Data API Configuration ---
FINANCIAL_DATA_API_KEY = get_env_var("FINANCIAL_DATA_API_KEY")
FINANCIAL_DATA_API_BASE_URL = get_env_var("FINANCIAL_DATA_API_BASE_URL")

# --- News API Configuration ---
NEWS_API_KEY = get_env_var("NEWS_API_KEY")
NEWS_API_BASE_URL = get_env_var("NEWS_API_BASE_URL")

# --- LLM Provider API Keys ---
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
GROQ_API_KEY = get_env_var("GROQ_API_KEY")
ANTHROPIC_API_KEY = get_env_var("ANTHROPIC_API_KEY") # Example

# --- Trading Parameters ---
# For STOCKS_TO_MONITOR, provide a default list if the variable is not set or empty.
DEFAULT_STOCKS = "AAPL,MSFT,GOOG"
STOCKS_TO_MONITOR = get_list_env_var("STOCKS_TO_MONITOR", DEFAULT_STOCKS)
if not STOCKS_TO_MONITOR: # If env var was empty string leading to empty list
    logger.info(f"STOCKS_TO_MONITOR was empty, using default: {DEFAULT_STOCKS}")
    STOCKS_TO_MONITOR = [item.strip() for item in DEFAULT_STOCKS.split(',') if item.strip()]


# DEFAULT_RISK_LEVEL, ensure it's a float
raw_risk_level = get_env_var("DEFAULT_RISK_LEVEL", "0.01")
try:
    DEFAULT_RISK_LEVEL = float(raw_risk_level)
except ValueError:
    logger.warning(f"Invalid value for DEFAULT_RISK_LEVEL: '{raw_risk_level}'. Using default 0.01.")
    DEFAULT_RISK_LEVEL = 0.01

# --- Logging Configuration ---
LOG_LEVEL = get_env_var("LOG_LEVEL", "INFO").upper()
LOG_FILE = get_env_var("LOG_FILE", "app.log") # Path to log file, can be empty
if LOG_FILE == "": # Explicitly handle empty string as None for logger setup
    LOG_FILE = None 

# --- Example of how to use these variables elsewhere ---
# from contrarian_trader.src.utils.config import FINANCIAL_API_KEY, NEWS_API_KEY, STOCKS_TO_MONITOR

if __name__ == '__main__':
    # This section runs when the script is executed directly, for testing.
    print("--- Configuration Test ---")
    print(f"Financial API Key: {FINANCIAL_DATA_API_KEY}")
    print(f"Financial API Base URL: {FINANCIAL_DATA_API_BASE_URL}")
    print(f"News API Key: {NEWS_API_KEY}")
    print(f"News API Base URL: {NEWS_API_BASE_URL}")
    print(f"OpenAI API Key: {OPENAI_API_KEY}")
    print(f"Groq API Key: {GROQ_API_KEY}")
    print(f"Stocks to Monitor: {STOCKS_TO_MONITOR} (Type: {type(STOCKS_TO_MONITOR)})")
    print(f"Default Risk Level: {DEFAULT_RISK_LEVEL} (Type: {type(DEFAULT_RISK_LEVEL)})")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Log File: {LOG_FILE if LOG_FILE is not None else 'Console only'}")

    # Test get_env_var with a non-existent variable and a default
    print(f"Test NonExistentVar (default 'TestDefault'): {get_env_var('NON_EXISTENT_VAR_TEST', 'TestDefault')}")
    # Test get_env_var with a non-existent variable and no default
    print(f"Test NonExistentVar (no default): {get_env_var('NON_EXISTENT_VAR_NO_DEFAULT_TEST')}")

    # To actually test loading from .env, you would create a .env file in the root
    # (e.g., contrarian_trader/.env) with some of these values and run this script.
    # Example .env content:
    # FINANCIAL_DATA_API_KEY="actual_financial_key_from_env"
    # STOCKS_TO_MONITOR="TSLA,NVDA"
    print("\nTo fully test, create a .env file in the project root and set some variables.")
    print("For example, set STOCKS_TO_MONITOR=\"TSLA,NVDA\" in .env")
    print("Then run `python -m contrarian_trader.src.utils.config` from the project root.")
