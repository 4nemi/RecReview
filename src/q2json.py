import re
import csv
import json
import pandas as pd
#.csvから.jsonに変換

def csv_to_json(csvFilePath, jsonFilePath):

    with open(csvFilePath, encoding="utf-8") as csvf:
        reader = csv.reader(csvf)
        json_dict = {}
        for i, rows in enumerate(reader):
            if i == 0:
                continue
            key = rows[1]
            rows = rows[2:]
            sub_dict = {}
            for i in range(0, len(rows)-2, 3):
                if rows[i] == "":
                    break
                #sub_key = re.sub(r"\s", "", rows[i])
                sub_dict[rows[i]] = float(rows[i+2])
            json_dict[key] = sub_dict

    with open(jsonFilePath, 'w', encoding="utf-8") as jsonf:
        jsonf.write(json.dumps(json_dict, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    csvFilePath = "../input/questionnaire_ans.csv"
    jsonFilePath = "../input/questionnaire_ans.json"

    csv_to_json(csvFilePath, jsonFilePath)