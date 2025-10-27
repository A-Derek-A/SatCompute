import click
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",  # 只显示INFO及以上level的
    format="[%(module)s.%(funcName)s] %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
)

logger = logging.getLogger("rich")
