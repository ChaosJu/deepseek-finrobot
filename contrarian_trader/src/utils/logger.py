import logging
import sys

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logger(name: str = 'contrarian_trader', 
                 log_level: str = 'INFO', 
                 log_file: str = None,
                 log_format: str = DEFAULT_LOG_FORMAT,
                 date_format: str = DEFAULT_DATE_FORMAT) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name: The name of the logger.
        log_level: The logging level (e.g., 'INFO', 'DEBUG').
        log_file: Optional path to a file to save logs. If None, only logs to console.
        log_format: The format string for log messages.
        date_format: The format string for dates in log messages.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent multiple handlers if logger already configured (e.g., in interactive sessions)
    if logger.handlers:
        return logger # Or clear handlers: logger.handlers = [] 

    level = logging.getLevelName(log_level.upper())
    logger.setLevel(level)

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}", exc_info=True)

    logger.info(f"Logger '{name}' configured with level {log_level.upper()}.")
    # Test messages
    # logger.debug("This is a debug message.")
    # logger.info("This is an info message.")
    # logger.warning("This is a warning message.")
    # logger.error("This is an error message.")
    # logger.critical("This is a critical message.")
    
    return logger

if __name__ == '__main__':
    # Example Usage:
    
    # Basic console logger
    logger1 = setup_logger(name='MyConsoleApp', log_level='DEBUG')
    logger1.info("Console logger test successful.")
    logger1.debug("This is a debug message for console logger.")

    # Logger with file output
    try:
        logger2 = setup_logger(name='MyFileApp', log_level='INFO', log_file='app.log')
        logger2.info("File logger test: This should go to console and app.log.")
        logger2.warning("File logger test: A warning message.")
    except Exception as e:
        print(f"Error setting up logger2: {e}") # Should be caught by logger itself if it's critical

    # Test re-configuring (should return existing logger without adding more handlers)
    logger1_again = setup_logger(name='MyConsoleApp', log_level='INFO')
    if len(logger1_again.handlers) > 1: # Basic check, console handler is 1 if no file
         # If file logging was added to MyConsoleApp, it could be > 1
         # A more robust check would be type of handlers or their names
        print(f"Warning: Logger 'MyConsoleApp' might have duplicate handlers. Count: {len(logger1_again.handlers)}")
    logger1_again.info("Testing logger retrieval - should not reconfigure excessively.")
    
    print("Logger examples complete. Check 'app.log' if created.")
