#!/usr/bin/bash
kaggle competitions download jigsaw-toxic-comment-classification-challenge
mv train.csv.zip ../../data/raw/train.csv.zip
mv test.csv.zip ../../data/raw/test.csv.zip
mv test_labels.csv.zip ../../data/raw/test_labels.csv.zip
mv sample_submission.csv.zip ../../data/raw/sample_submission.csv.zip
cd ../../data/raw
unzip train.csv.zip -d ./
unzip test.csv.zip -d ./
unzip test_labels.csv.zip -d ./
unzip sample_submission.csv.zip -d ./
