#!/bin/bash

# Setting parameters
ALGORITHM=${1:-"fed_tld"}
CLIENT_NUM=5
R=5

METRIC_TYPE="/gemini/code/evaluate/metrics/f1"
DATA_DIR="/gemini/data-1/non-iid-original-data"
CENTRAL_MODEL="/gemini/code/model/roberta-large"
LOCAL_MODELS="/gemini/code/model/bert-base-cased,/gemini/code/model/bert-large-cased,/gemini/code/model/roberta-base,/gemini/code/model/roberta-large,/gemini/code/model/xlnet-large-cased"
OUTPUT_DIR="/gemini/output"

PUBLIC_RATIO=0.2
BATCH_SIZE=32
MAX_SEQ_LEN=128
WEIGHT_DECAY=0
LR=2e-5
EPOCHS=3
DIS_EPOCHS=3

# Run
python /gemini/code/main.py \
    --algorithm $ALGORITHM \
    --K $CLIENT_NUM \
    --R $R \
    --metric_type $METRIC_TYPE \
    --data_dir $DATA_DIR \
    --central_model $CENTRAL_MODEL \
    --local_models $LOCAL_MODELS \
    --output_dir $OUTPUT_DIR \
    --public_ratio $PUBLIC_RATIO \
    --batch_size $BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LEN \
    --weight_decay $WEIGHT_DECAY \
    --lr $LR \
    --E $EPOCHS \
    --dis_epochs $DIS_EPOCHS \
