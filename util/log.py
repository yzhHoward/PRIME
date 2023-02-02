import os
import sys
import logging


def init_logging(log_root, models_root=None):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if models_root is not None:
        handler_file = logging.FileHandler(
            os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)
        log_root.addHandler(handler_file)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_stream)
