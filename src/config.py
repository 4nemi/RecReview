from transformers import AutoTokenizer, AutoConfig
PARTICIPANT = "亀山直生"
MODEL_NUM = 1
TEST_NAME = '空の中 (角川文庫)'
TRAIN_FILE = '../input/review1_cleaned.csv'

DEBUG = True
EPOCHS = 10
if DEBUG:
    EPOCHS = 2

MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"

MODEL_PATH = "../models/f'model_{MODEL_NUM}.bin'"

MAX_LEN = 512

BATCH_SIZE = 16

ENCODER_LR = 2e-5
DECODER_LR = 2e-5

PRINT_FREQ = 50

TOKENIZER = AutoTokenizer.from_pretrained(MODEL)

SEED = 42