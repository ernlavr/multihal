#!/bin/bash

# Some datasets are not available from HuggingFace, therefore we need to download them from the original source
# (last accessed 04.01.2025)

# Create a 'res' directory if it doesnt exist
if [ ! -d "res" ]; then
  mkdir res
fi

# Download the dataset from the original source

# SimpleQA: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
mkdir -p res/simpleqa
wget https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv -O res/simpleqa/simple_qa_test_set.csv

# Shroom2024 https://helsinki-nlp.github.io/shroom/2024
wget 'https://drive.usercontent.google.com/download?id=1wlGZL8Sdqu7xZngcUSrDqp3DCSkYWoaE&export=download&authuser=0' -O shroom2024.zip # replace
mkdir -p res/shroom2024
unzip -o shroom2024.zip -d res/shroom2024
rm shroom2024.zip

# Shroom2025 https://helsinki-nlp.github.io/shroom/
wget 'https://a3s.fi/mickusti-2007780-pub/train.zip' -O shroom2025.zip
mkdir -p res/shroom2025
unzip -o shroom2025.zip -d res/shroom2025
rm shroom2025.zip

# get DefAn https://github.com/ernlavr/DefAn
mkdir -p res/defan
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_1_public.csv' -O res/defan/QA_domain_1_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_2_public.csv' -O res/defan/QA_domain_2_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_3_public.csv' -O res/defan/QA_domain_3_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_4_public.csv' -O res/defan/QA_domain_4_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_5_public.csv' -O res/defan/QA_domain_5_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_6_public.csv' -O res/defan/QA_domain_6_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_7_public.csv' -O res/defan/QA_domain_7_public.csv
wget 'https://raw.githubusercontent.com/ernlavr/DefAn/refs/heads/main/DefAn-public/QA_domain_8_public.csv' -O res/defan/QA_domain_8_public.csv