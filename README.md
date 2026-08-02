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
HisPos/streamlit_app/data/test.tsv
```

When the two tags (POS1, POS2) disagree, then the row will be marked with a MISMATCH tag. This makes it easy to identify potential errors and to correct them in an efficient manner.

You can either select the correct tag by clicking on one of the checkboxes or you can insert the tag by replacing the MISMATCH tag with the PoS label.

You can use the ```Previous/Next``` buttons to navigate or you can insert the sentence number in the text field under ```Jump to sentence``` and press ENTER.

You can save the file, using the ```Save``` button. The saved file has an additional column ```GOLDPOS``` with the corrected PoS tag.
You can also upload the saved file and continue with the error correction.
 

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit) - see the LICENSE file for details.
 