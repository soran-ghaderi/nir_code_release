import logging
import random

import numpy as np
import torch
from rich.logging import RichHandler

import configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logger():
    if configs.USE_RICH:

        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
        logger = logging.getLogger("rich")
    else:

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    return logger
