import pandas as pd
import json
import sys
sys.dont_write_bytecode = True
import config

if __name__ == "__main__":
    #jsonファイルの読み込み
    with open("../input/questionnaire_ans.json", 'r', encoding="utf-8") as f:
        json_dict = json.load(f)
    ans_dict = json_dict[config.PARTICIPANT]
    df = pd.read_csv(config.TRAIN_FILE)
    #target列にrating-ans_dict['name']を追加
    df['p_rating'] = df['name'].apply(lambda x: ans_dict[x])
    df['target'] = df['p_rating'] - df['rating']
    #p_rating列を削除
    df = df.drop('p_rating', axis=1)
    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    df['test'] = -1
    df.loc[df.name == config.TEST_NAME, 'test'] = 1
    df.to_csv(config.TRAIN_FILE, index=False)