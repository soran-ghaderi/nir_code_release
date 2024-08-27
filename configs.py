MAX_LENGTH = 128
# BATCH_SIZE = 5
# NUM_CONTEXTS = 10
CRV_DIM = 4096  # LLaMA hidden size -> should derive it from the model configs
SUBSET_SIZE = 7
CRV_SAVE_BATCH = 20
CRV_LAYERS = [
    1,
    10,
    15,
    20,
    32,
]  # the hidden states include the input embedding layer therefore idx 0 is the embeds lyr
