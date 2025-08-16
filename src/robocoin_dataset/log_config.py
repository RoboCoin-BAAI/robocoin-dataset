from dataclasses import dataclass


@dataclass()
class LogConfig:
    """
    Configuration class for logging settings.

    This class defines the configuration parameters for logging in the application,
    including log directory, console output options, and log level.

    Attributes:
        log_dir (str): Directory path where log files will be stored. Defaults to empty string.
        log_to_console (bool): Flag indicating whether logs should be output to console. Defaults to True.
        log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR"). Defaults to "INFO".
    """

    log_dir: str = ""
    log_to_console: bool = True
    log_level: str = "INFO"
