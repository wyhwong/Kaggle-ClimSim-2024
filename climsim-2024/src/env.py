import os


RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2024"))

# For logger
STREAMING_LOG_LEVEL = int(os.getenv("STREAMING_LOG_LEVEL", "30"))
FILE_LOG_LEVEL = int(os.getenv("FILE_LOG_LEVEL", "20"))
LOG_FILEPATH = os.getenv("LOG_FILEPATH", "./runtime.log")
# NOTE: Although LOG_FMT and LOG_DATEFMT are in env.py, we do not expect
#       them to be changed by environment variables. They define the logging
#       style of analyticks and should not be changed.
LOG_FMT = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"

DEVICE = os.getenv("DEVICE", "cuda")

# For hyperparameters (LR scheduler)
INITIAL_LR = float(os.getenv("INITIAL_LR", "1e-4"))
MAXIMUM_LR = float(os.getenv("MAXIMUM_LR", "1e-3"))
N_WARMUP_EPOCHS = int(os.getenv("N_WARMUP_EPOCHS", "1"))
N_DECAY_EPOCHS = int(os.getenv("N_DECAY_EPOCHS", "22"))
ALPHA = float(os.getenv("ALPHA", "0.1"))

# For data module
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2048"))
# BUFFER_SIZE: number of batches being preloaded in memory
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "100"))
# N_GROUP_PER_SAMPLING: number of groups being sampled in each iteration
N_GROUP_PER_SAMPLING = int(os.getenv("N_GROUP_PER_SAMPLING", "3"))
# N_BATCH_PER_SAMPLING: number of batches being sampled in each iteration
N_BATCH_PER_SAMPLING = int(os.getenv("N_BATCH_PER_SAMPLING", "100"))
IS_NORMALIZED = bool(os.getenv("IS_NORMALIZED", "False"))
IS_STANDARDIZED = bool(os.getenv("IS_STANDARDIZED", "True"))

# For trainer
N_EPOCHS = int(os.getenv("N_EPOCHS", "25"))
MAXIMUM_TRAINING_TIME_IN_HOUR = int(os.getenv("MAXIMUM_TRAINING_TIME_IN_HOUR", "72"))
