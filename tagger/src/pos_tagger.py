# -*- coding: utf-8 -*-
import os, sys
import torch
import pandas as pd
import numpy as np
import evaluate
import random
import utils
from transformers import set_seed
from transformers import BertTokenizerFast, BertForTokenClassification
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
Basic POS tagger for historical texts.

Adapted from https://huggingface.co/learn/llm-course/en/chapter7/2
""" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(random_seed):
    set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    logging.info("Random seed set as {random_seed}")



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

    SEED = int(config_parser['PARAM']['seed'])
    seed_all(SEED)
    EVAL_ON_DEV = config_parser['BASE'].getboolean('eval_on_dev')
    DATA_DIR = config_parser['DATA']['data_dir']
    RESULT_DIR  = config_parser['DATA']['result_dir'] 
    MODEL_DIR   = config_parser['MODEL']['model_dir']
    MODEL_NAME  = config_parser['MODEL']['model_name']
    MODEL_ABBR  =  config_parser['MODEL']['model_abbr']

    """ Define checkpoint and hyperparameters """
    SAVED_MODEL_PATH = config_parser['MODEL']['model_dir']
    LEARNING_RATE = float(config_parser['PARAM']['lr'])
    EPOCHS = int(config_parser['PARAM']['epochs'])
    BATCH_SIZE = int(config_parser['PARAM']['batch_size'])


    # Read training data:
    train_df = pd.read_csv(DATA_DIR + "/train.tsv", sep="\t", header=None, names=["token_id", "token", "label"], quoting=3, na_filter=False)
    dev_df = pd.read_csv(DATA_DIR + "/dev.tsv", sep="\t", header=None, names=["token_id", "token", "label"], quoting=3, na_filter=False)
    test_df = pd.read_csv(DATA_DIR + "/test.tsv", sep="\t", header=None, names=["token_id", "token", "label"], quoting=3, na_filter=False)
 
    train_df["label"] = train_df["label"].replace("NONE", "O")
    dev_df["label"] = dev_df["label"].replace("NONE", "O")
    test_df["label"] = test_df["label"].replace("NONE", "O")
    logging.info("Data lenght: train {len(train_df)}, dev {len(dev_df)}, test {len(test_df)}")

    dfs_train = [df.reset_index(drop=True) for _, df in train_df.groupby(train_df['token_id'].eq(1).cumsum())]
    dfs_val = [df.reset_index(drop=True) for _, df in dev_df.groupby(dev_df['token_id'].eq(1).cumsum())]
    dfs_test = [df.reset_index(drop=True) for _, df in test_df.groupby(test_df['token_id'].eq(1).cumsum())]
    logging.info("Number of sentences in train: {len(dfs_train)}")


    # Convert data into one sentence-per-row format:
    sentences_train = utils.one_sentence_per_row(dfs_train)
    sentences_val = utils.one_sentence_per_row(dfs_val)
    sentences_test = utils.one_sentence_per_row(dfs_test) 

    train_ds = Dataset.from_pandas(sentences_train, split="train")
    val_ds = Dataset.from_pandas(sentences_val, split="validation")
    test_ds = Dataset.from_pandas(sentences_test, split="test")
    
    # Specify POS labels
    unique_labels = ['$(', '$,', '$.', 'ADJA', 'ADJD', 'ADV', 'APPO', 'APPR', 'APPRART', 'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOKOM', 'KON', 'KOUI', 'KOUS', 'NE', 'NN', 'PAV', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPER', 'PPOSAT', 'PRELAT', 'PRELS', 'PRF', 'PTKA', 'PTKANT', 'PTKNEG', 'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP', 'XY', 'PPOSS', 'VMPP']   

    dta_dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds })

    # Define label mappings for BERT:
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
    logging.info("Label-id mappings:\n {labels_to_ids}")

    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenised_dataset = tokenise_dataset(dta_dataset)
 
    data_collator = DataCollatorForTokenClassification(tokenizer=tokeniser)

    accuracy = evaluate.load("accuracy")

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        num_train_epochs=EPOCHS,
        weight_decay=0.01, # default in AdamW
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",
        optim="adamw_torch", 
        metric_for_best_model="accuracy"
    )
   
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_labels), id2label=ids_to_labels, label2id=labels_to_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset["train"],
        eval_dataset=tokenised_dataset["validation"],
        processing_class=tokeniser,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info("\nTraining hyperparameters:")
    logging.info("BERT checkpoint: {}".format(MODEL_NAME))
    logging.info("Epochs: {}".format(EPOCHS))
    logging.info("Learning rate: {}".format(LEARNING_RATE))
    logging.info("Batch size: {}\n".format(BATCH_SIZE))

    trainer.train()
    trainer.save_model()

    # EVALUATION  
    val_dataset = tokenised_dataset["validation"] 
    test_dataset = tokenised_dataset["test"]
    
    # Do we want to test on the dev set or on test?
    # Default: evaluation on development set
    if EVAL_ON_DEV:
        predictions_raw = trainer.predict(val_dataset)
    else:
        predictions_raw = trainer.predict(test_dataset)

    predictions_per_sentence = []
    gold_per_sentence = []
    for i in range(predictions_raw.predictions.shape[0]):
        logits_clean = predictions_raw.predictions[i][predictions_raw.label_ids[i] != -100]
        label_clean = predictions_raw.label_ids[i][predictions_raw.label_ids[i] != -100]
        predictions = np.argmax(logits_clean, axis=-1)
        predictions_per_sentence.append(predictions)
        gold_per_sentence.append(label_clean)
    
    
    pred_labels = [[ids_to_labels[k] for k in sentence] for sentence in predictions_per_sentence] 
 
    if EVAL_ON_DEV:
        df_eval_predictions = sentences_val.copy(deep=True)
    else:
        df_eval_predictions = sentences_test.copy(deep=True)
    
    df_eval_predictions["predictions"] = pred_labels
    pd.set_option('display.max_columns', None)
    df_predictions_exploded = df_eval_predictions.explode(df_eval_predictions.columns.tolist())
    
    
    # Save predictions:
    if EVAL_ON_DEV:
        suffix = "/predictions_dev_set.tsv"
    else:
        suffix = "/predictions_test_set.tsv"
    pred_file = RESULT_DIR + suffix

    with open(pred_file, "w") as outf:
        df_predictions_exploded.to_csv(outf, sep="\t")

