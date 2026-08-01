# -*- coding: utf-8 -*-
import os, sys
import glob
import torch
import pandas as pd
import numpy as np
import evaluate
import random
import utils
from transformers import BertForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score 
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW 
import configparser
import logger
import logging


"""
Simple POS tagger for historical texts.

Adapted from https://huggingface.co/learn/llm-course/en/chapter7/2

Please change the values of the capitalized variables below depending on how you wish to use this script.
""" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def tokenise_and_align_labels(examples):
    tokenised_inputs = tokeniser(examples["tokens"], truncation=True,  is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"goldlabels"]): 
        # Map tokens to their respective word
        word_ids = tokenised_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        # Set the special tokens to -100.
        for word_idx in word_ids:  
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:  
                label_ids.append(labels_to_ids[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenised_inputs["labels"] = labels
    return tokenised_inputs


def tokenise_dataset(dta_dataset):
    # Define label mappings for BERT:
    tokenised_datased = dta_dataset.map(tokenise_and_align_labels, batched=True)
    return tokenised_datased
 


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [unique_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [unique_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = accuracy_score(true_labels[0], true_predictions[0])
    return {
        "accuracy": results,
    }



if __name__ == '__main__':

    ###############################
    # Read config and set seed
    config_parser = configparser.ConfigParser()
    config_parser.read(sys.argv[1])

    logging.basicConfig(level=logging.INFO)
    
    DATA_DIR    = config_parser['DATA']['data_dir']
    RESULT_DIR  = config_parser['DATA']['result_dir'] 
    MODEL_DIR   = config_parser['MODEL']['model_dir']
    MODEL_NAME  = config_parser['MODEL']['model_name']
    MODEL_ABBR  =  config_parser['MODEL']['model_abbr']

    BATCH_SIZE = int(config_parser['PARAM']['batch_size'])

    # Read all files in DATA_DIR
    infiles = glob.glob(DATA_DIR + '/*.tsv')
    
    for f in infiles:
        logging.info("Reading input file {}.".format(f))
        test_df = pd.read_csv(f, sep="\t", header=None, names=["token_id", "token", "label"], quoting=3, na_filter=False)
 
        test_df["label"] = test_df["label"].replace("NONE", "O")

        dfs_test = [df.reset_index(drop=True) for _, df in test_df.groupby(test_df['token_id'].eq(1).cumsum())]

        # Turn dataset into one sentence per row:
        sentences_test = utils.one_sentence_per_row(dfs_test) 
        test_ds = Dataset.from_pandas(sentences_test, split="test")
    
        # Specify POS labels
        unique_labels = ['$(', '$,', '$.', 'ADJA', 'ADJD', 'ADV', 'APPO', 'APPR', 'APPRART', 'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOKOM', 'KON', 'KOUI', 'KOUS', 'NE', 'NN', 'PAV', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPER', 'PPOSAT', 'PRELAT', 'PRELS', 'PRF', 'PTKA', 'PTKANT', 'PTKNEG', 'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP', 'XY', 'PPOSS', 'VMPP']   

        pos_dataset = DatasetDict({
            "test": test_ds })

        # Define label mappings for BERT:
        labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
        ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

        tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenised_dataset = tokenise_dataset(pos_dataset)
 
        data_collator = DataCollatorForTokenClassification(tokenizer=tokeniser)

        accuracy = evaluate.load("accuracy")

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            push_to_hub=False,
            report_to="none",
        ) # linear scheduler is default

        logging.info("\nPredict labels...")
        eval_dataset = tokenised_dataset["test"]
        
        logging.info("Loading model from {}".format(MODEL_DIR))
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, num_labels=len(unique_labels), id2label=ids_to_labels, label2id=labels_to_ids)
        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokeniser,
            data_collator=data_collator,
        )
        predictions_raw = trainer.predict(eval_dataset)
        

        gold_per_sentence, predictions_per_sentence = [], []
        for i in range(predictions_raw.predictions.shape[0]):
            logits_clean = predictions_raw.predictions[i][predictions_raw.label_ids[i] != -100]
            label_clean = predictions_raw.label_ids[i][predictions_raw.label_ids[i] != -100]
            predictions = np.argmax(logits_clean, axis=-1)
            predictions_per_sentence.append(predictions)
            gold_per_sentence.append(label_clean)
    
    
        predictions_textual_labels = [[ids_to_labels[k] for k in sentence] for sentence in predictions_per_sentence]

        df_eval_predictions = sentences_test.copy(deep=True)
        df_eval_predictions["predictions"] = predictions_textual_labels
    
        pd.set_option('display.max_columns', None)
        
        df_test_predictions_exploded = df_eval_predictions.explode(df_eval_predictions.columns.tolist())
        pred_file = RESULT_DIR + '/' + f.split('/')[-1]

        # Write predictions to file
        logging.info("Writing predictions to {}.".format(pred_file))
        with open(pred_file, "w") as outf:
            df_test_predictions_exploded.to_csv(outf, sep="\t")

