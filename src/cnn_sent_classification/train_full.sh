#!/usr/bin/bash
python main.py -path ../../data/raw -csv train.csv -test_csv test.csv -test_labels ../../data/raw/test_labels.csv -model_name SimpleCNN -result_dir ./results/ -sub_path ./submissions/simple_cnn_sd_0.044.csv -exp_name simple_cnn
