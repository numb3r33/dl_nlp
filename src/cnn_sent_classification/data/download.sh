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
curl 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip' -H 'authority: dl.fbaipublicfiles.com' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8' -H 'referer: https://fasttext.cc/docs/en/english-vectors.html' -H 'accept-encoding: gzip, deflate, br' -H 'accept-language: en-US,en;q=0.9' -H 'cookie: __cfduid=dd28d6dd54a63a402756a4185637c96f51553678600' --compressed -o ../../data/processed/crawl-300d-2M.vec.zip
cd ../../data/processed
unzip crawl-300d-2M.vec.zip -d ./
