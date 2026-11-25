import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent / "resep_rag.log"


def setup_logging(console: bool = True) -> None:
    """Write log messages to resep_rag.log (appended on every run).

    console=False keeps the terminal clean, e.g. for the interactive chat in agent.py.
    """
    handlers: list[logging.Handler] = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
