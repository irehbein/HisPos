import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys

"""
GOLDPOS SID     TID     TOKEN   POS1    POS2
_       2       1       gestellt        VVPP    VVPP
_       2       2       durch   APPR    APPR
NN      2       3       Medikus NE      NN
NE      3       1       Joannem FM      NE
"""
def read_gold(gfile):
    df = pd.read_csv(gfile, sep="\t", header=0)
    df['PRED'] = ["_" for x in list(df.SID)]
    return df

"""
{"sid": 2, "token_ids": [1, 2, 3], "token": ["gestellt", "durch", "Medikus"], "pos": ["VVPP", "APPR", "NE"], "error": ["OK", "OK", "NN"], "gold": ["_", "_", "NN"], "error_ids": [2], "predictions": ["NN"]}
"""
def read_pred(pfile):
    data = []
    with open(pfile, "r") as inf:
        for line in inf:
            data.append(json.loads(line.strip()))
    return data


def add_pred_to_gold(gold_df, pred_list):

    for item in pred_list:
        err_ids = list(item["error_ids"])
        preds   = list(item["predictions"])
        for i in range(len(err_ids)):
            gold_df.loc[(gold_df['SID'] == item['sid']) & (gold_df['TID'] == err_ids[i]+1), 'PRED'] = preds[i]


def acc(gold, pred):
    print(accuracy_score(gold, pred))

def classification_report(gold, pred):
    print(classification_report(gold, pred, weighted='micro'))


def evaluate(df):

    err_tags = ['NN', 'NE']
    gold_list = list(df.GOLDPOS)
    pos1_list = list(df.POS1)
    pos2_list = list(df.POS2)
    pred_list = list(df.PRED)

    gold, pos1, pos2, pred = [], [], [], []

    for i in range(len(gold_list)):
        if gold_list[i] != "_" and pos1_list[i] in err_tags and pos2_list[i] in err_tags:
            gold.append(gold_list[i])
            pos1.append(pos1_list[i])
            pos2.append(pos2_list[i])
            pred.append(pred_list[i])

    print("LENGTH", len(gold))
    print(gold[0:5])
    print(pos1[0:5])
    print(pos2[0:5])
    print(pred[0:5])
    print()    



    print("Accuracy DTA:")
    acc(gold, pos1)
    print("Accuracy ENSEMBLE:")
    acc(gold, pos2)
    print("Accuracy LLM:")
    acc(gold, pred)



goldfile = sys.argv[1] # e.g., 'tagger_output/gold_Brendel.tsv'
predfile = sys.argv[2] # e.g., predictions/NN_NE_ERROR_gold_Brendel.tsv


gold_df = read_gold(goldfile)
pred_list = read_pred(predfile)

add_pred_to_gold(gold_df, pred_list)

gold_df.to_csv('out.csv', sep="\t")

evaluate(gold_df)