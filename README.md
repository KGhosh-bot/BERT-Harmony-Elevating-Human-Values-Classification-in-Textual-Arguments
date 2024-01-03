# Human Value Detection
This project addresses the Human Value Detection Challenge, where the objective is to classify, given a textual argument and a human value category, classify whether or not the argument draws on that category. 
Human values behind natural language arguments, such as to have 'freedom of thought' or to be 'broad-minded' are commonly accepted answers and logic to why something is desirable in the ethical sense and are thus essential both in real world argumentation and theoretical argumentation frameworks. The goal is to perform automatic multi label classification using several neural models considering solely level 3 value categories. The experimentation achieved a maximum F1-score of 0.88 and an average of 0.77.

## Problem definition

Arguments are paired with their conveyed human values.
Arguments are in the form of **premise** $\rightarrow$ **conclusion**.

### Example:

**Premise**: *``fast food should be banned because it is really bad for your health and is costly''*

**Conclusion**: *``We should ban fast food''*

**Stance**: *in favour of*

<p align="center">
    <img src="images/human_values.png" alt="human values", style="width: 400px; height: 400px;"/></center>
</p>

## Corpus

The official page of the challenge [here](https://touche.webis.de/semeval23/touche23-web/) offers several corpora for evaluation and testing.

I worked with the standard training, validation, and test splits.

#### Arguments
* arguments-training.tsv
* arguments-validation.tsv
* arguments-test.tsv

#### Human values
* labels-training.tsv
* labels-validation.tsv
* labels-test.tsv

### Annotations

To address a multi-label classification problem, I consider **level 3** categories:

* Openness to change
* Self-enhancement
* Conversation
* Self-transcendence

### Models

* **Baseline**: a random uniform classifier (an individual classifier per category).
* **Baseline**: a majority classifier (an individual classifier per category).

<br/>

* **BERT w/ C**: a BERT-based classifier that receives an argument **conclusion** as input.
* **BERT w/ CP**: added argument **premise** as an additional input.
* **BERT w/ CPS**: added argument premise-to-conclusion **stance** as an additional input.

## Flow of the notebook
The notebook will be divided into seperate sections to provide a organized walk through the process used. The sections are:

1. Importing Python Libraries and preparing the environment
2. Importing and Pre-Processing the domain data
3. Preparing the Dataset suitable for BERT
4. Fine Tuning the Model
5. Training and Validating the Model Performance on the trained Model for three different seeds
6. Predicting on Test set for three different seeds
7. Comparing the different models and their variants
8. Analysing for errors on the best model
