import random

import numpy
import torch


def reproducible_worker_init_fn(worker_id: int) -> None:
    """
    see discussion at https://github.com/pytorch/pytorch/issues/5059
    and the article at https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
