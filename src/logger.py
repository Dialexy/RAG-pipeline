"""
Centralised logging setup. Import get_logger and call it once per module.
"""

import logging
import transformers
from tqdm import tqdm as _tqdm

transformers.logging.set_verbosity_error()

# safetensors uses tqdm directly when loading weights; patch __init__ to force disable
_orig_tqdm_init = _tqdm.__init__

def _silent_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _orig_tqdm_init(self, *args, **kwargs)

_tqdm.__init__ = _silent_tqdm_init


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a StreamHandler, configured once."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
