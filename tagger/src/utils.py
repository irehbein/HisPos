# -*- coding: utf-8 -*-
import pandas as pd 


def one_sentence_per_row(dfs):
    sentence_dicts = []
    for df in dfs:
        token_ids = df.token_id.tolist()
        tokens = df.token.tolist()
        labels = df.label.tolist()
        sentence_dict = {
            'token_ids': token_ids,
            'tokens': tokens, 
            'goldlabels': labels}
        sentence_dicts.append(sentence_dict)
    sentences = pd.DataFrame(sentence_dicts) 

    return sentences
    



