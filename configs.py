import logging

MAX_LENGTH = 512
# BATCH_SIZE = 5
# NUM_CONTEXTS = 10
CRV_DIM = 4096  # LLaMA hidden size -> should derive it from the model configs
SUBSET_SIZE = 7
CRV_SAVE_BATCH = 20
# CRV_LAYERS = [
#     1,
#     5,
#     10,
#     15,
#     20,
#     32,
# ]  # the hidden states include the input embedding layer therefore idx 0 is the embeds lyr

CRV_LAYERS = [23, 1, 10, "orig"]
USE_RICH = False
logging_level = logging.ERROR
