from transformers import AutoTokenizer, AutoConfig
PARTICIPANT = ""
MODEL_NUM = 1
EXPRRIMENT = 'kfold' #stratified kfold or group kfold or holdout
TARGET = 'p_rating and diff' #rating or diff
TEST_NAME = []
TRAIN_FILE = '../input/review1_cleaned_2.csv'

DEBUG = False
EPOCHS = 10
TRN_FOLD = [0, 1, 2, 3]
if DEBUG:
    EPOCHS = 2
    TRN_FOLD = [0]
    
ALPHA = 0.5

WANDB = True

MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"

MODEL_PATH = "../models/"
OUTPUT_PATH = "../output/"

MAX_LEN = 512

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16

ENCODER_LR = 2e-5
DECODER_LR = 2e-5

DROP_OUT = 0.2

WEIGHT_DECAY = 0.01

PRINT_FREQ = 50

TOKENIZER = AutoTokenizer.from_pretrained(MODEL)

SEED = 42