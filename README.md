# HisPos: Automatic Error Correction for PoS in Historical Text

This repository contains the code for the PoS tagger and the streamlit app for PoS error correction used in our paper:

```
@inproceedings{rehbein-etal-2026-HisPos,
    title = {Automatic Error Correction for PoS in Historical Text},
    author = {Ines Rehbein and Laura Duve and Antje Dammel},
    booktitle = {Proceedings of the 22th Conference on Natural Language Processing (KONVENS 2026)},
    address = {Hamburg, Germany},
    month = {sep},
    year = 2026,
}
```

## Description

In the paper, we focus on the problem of part-of-speech tagging of Early New High
German documents and evaluate methods to reduce human effort. Specifically, we present a pipeline and annotation interface that supports manual post-correction of automatically predicted tags and show that using open-weight LLMs for post-correction of PoS tags is feasible at least for some error types, when breaking
down the task into well-defined problems.

The repository includes: 
* the source code for the PoS tagger used to identify potential errors in the PoS predictions.
* the source code for prompting the LLMs to correct PoS errors in the data
* the streamlit app used to manually correct PoS errors, based on the predictions of the taggers.

<br/><br/>

## PoS Tagger

### Dependencies

See ```requirements.txt``` in the tagger folder.

### Data

For training, we use samples from the RIDGES corpus:

Lüdeling Anke, Odebrecht Carolin, Krause Thomas, Schnelle Gohar, Fischer Catharina (2020);
RIDGES Herbology (Version 9.0); Humboldt-Universität zu Berlin;
Homepage: http://korpling.org/ridges/;
DOI: https://doi.org/10.34644/laudatio-dev-PySSCnMB7CArCQ9CNKFY

For details how to obtain the data, please refer to the readme file in the folder ```tagger/data/```.

### Configuration 

Please check the files in the ```config``` folder and adapt the paths.

```
eval_on_dev         # use the file dev.tsv for evaluation

model_dir           # folder where the model will be saved

result_dir          # folder where the tagger output will be saved
```


### Training a PoS tagger

Change to the ```tagger``` folder and call the bash script ```run_train.sh```.


### Predicting PoS tags

Change to the ```tagger``` folder and call the bash script ```run_predict.sh```.


### Creating ensemble tags

We train a number of taggers on different samples from the RIDGES corpus (for details please refer to our paper). Then we take the majority vote from the tagger ensemble and add it as POS2 to our data (POS1 has been predicted by the [Cascaded Analysis Broker (CAB)](https://deutschestextarchiv.de/public/cab/)).
The so created files are input for the manual PoS Correction and for the Annotation Error Correction experiments (see below).

<br/><br/>

## Manual PoS Correction

The folder ```streamlit_app``` contains an annotation interface for PoS correction.

### Requirements

You need to install streamlit and pandas. We are using streamlit version 1.51.0 and pandas version 2.3.3.


### Start the app

Change to the ```streamlit_app``` folder and call:

```streamlit run app.py```

The app runs in the browser. To start, you need to upload a file that contains sentence ids, token ids, the word form, and two predicted PoS tags:

```
SID     TID     TOKEN   POS1    POS2
1       1       Nohthwendige    ADJA    ADJA
1       2       Vorsorge        NN      NN
1       3       Und     KON     KON
1       4       Anweisung       NN      NN
...     ...     ...             ...     ...
```

Please note that there are no newlines between sentences. For an example file, see:
```
HisPos/streamlit_app/data/input/test.tsv
```

When the two tags (POS1, POS2) disagree, then the row will be marked with a MISMATCH tag. This makes it easy to identify potential errors and to correct them in an efficient manner.

You can either select the correct tag by clicking on one of the checkboxes or you can insert the tag by replacing the MISMATCH tag with the PoS label.

You can use the ```Previous/Next``` buttons to navigate or you can insert the sentence number in the text field under ```Jump to sentence``` and press ENTER.

You can save the file, using the ```Save``` button. The saved file has an additional column ```GOLDPOS``` with the corrected PoS tag.
You can also upload the saved file and continue with the error correction.


### Manually corrected goldstandard

We release the manually corrected documents that we use for evaluating our Annotation Error Correction approach. You can find the four files in the folder ```HisPos/streamlit_app/data/gold/```.

Please note that this is a preliminary version of a more comprehensive dataset, which will be published once it is complete. The final version of the data may therefore differ from this version.

See here for more information on the project: 
[Referenzielle Praxis im Wandel: Das Pronomen man in der Diachronie des Deutschen](https://tp2.forschungsgruppe-pronomen.de/)



<br/><br/>

## Annotation Error Correction

### Prerequisites

You need to configure the following files:

* secrets.json
  * Contains your Huggingface API key and the HF_HOME environment variable (only needed when using local models) and your API key and URL for accessing the LLM (when using remote services).

* config/pos-llm-baseline.config
* config/pos-llm-correct.config
    * adapt model_name
    * adapt paths to input data (dataset_dir) and output folder (result_dir)

For automatic error correction, you also need to specify the prompt_type (```config/pos-llm-correct.config```). We provide templates for the following error types:
* FM_OTHER
* NE_OTHER
* VVFIN_OTHER
* PTKZU_OTHER
* ADV_OTHER
* GENERIC      # prompt the LLM to correct all tags marked as errors

For details, please refer to our paper.



### Baseline

Following a reviewer request, we added a baseline where we use an LLM as tagger to predict PoS tags for each wordform in our data in a zero-shot setting. Baseline accuracy for ```Gpt-oss-120B``` on our test set is 69.2. For comparison, the accuracy for the DTA predictions is 83.8 and the majority vote over the tags predicted by the ensemble tagger trained on RIDGES is 88.9.

Running the baseline:

```cd HisPos/aec

python ./src/pos_llm_baseline.py ./config/pos-llm-baseline.config
```

### Annotation Error Correction

You can run the error correction script by calling:

```
./src/pos_llm_correct.py ./config/pos-llm-correct.config
```


## License

This project is licensed under the [MIT License](https://opensource.org/license/mit) - see the LICENSE file for details.
 

