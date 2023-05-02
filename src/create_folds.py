import pandas as pd
#from sklearn import model_selection
import sys
sys.dont_write_bytecode = True
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_FILE)
    df = df.sample(frac=1).reset_index(drop=True)
    df['test'] = -1
    df.loc[df.name == config.TEST_NAME, 'test'] = 1
    df.to_csv(config.TRAIN_FILE, index=False)