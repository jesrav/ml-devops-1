import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)

# the handler determines where the logs go: stdout/file
shell_handler = RichHandler()

logger.setLevel(logging.INFO)
shell_handler.setLevel(logging.INFO)

# the formatter determines what our logs will look like
FMT_SHELL = "%(message)s"

shell_formatter = logging.Formatter(FMT_SHELL)

# here we hook everything together
shell_handler.setFormatter(shell_formatter)

logger.addHandler(shell_handler)
