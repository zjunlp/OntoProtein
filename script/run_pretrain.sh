#!/bin/bash

OUTPUT_DIR=data/output_data/filtered_ke_text
PRETRAIN_DATA_DIR=data/pretrain_data/pretrain_data_2021

# protein sequence setting
PROTEIN_DATA_FILE_NAME="swiss_seq"
IN_MEMORY=true
MAX_PROTEIN_SEQ_LENGTH=1024

# Go setting
USE_DESC=true
MAX_TEXT_SEQ_LENGTH=128
PROTEIN_GO_NUM_WORKERS=1
GO_GO_NUM_WORKERS=1
PROTEIN_SEQ_NUM_WORKERS=1

# negative sampling
NUM_PROTEIN_GO_NEG_SAMPLE=128
NUM_GO_GO_NEG_SAMPLE=128
NEGTIVE_SAMPLING_FN="simple_random"
PROTEIN_GO_SAMPLE_HEAD=false
PROTEIN_GO_SAMPLE_TAIL=true
GO_GO_SAMPLE_HEAD=true
GO_GO_SAMPLE_TAIL=true

# Protein sequence pretrained model
PROTEIN_MODEL_PATH='data/model_data/ProtBERT'

# OntoModel
TEXT_MODEL_PATH='data/model_data/OntoModel'
GO_ENCODER_CLS="bert"
PROTEIN_ENCODER_CLS="bert"
KE_EMBEDDING_SIZE=512
DOUBLE_ENTITY_EMBEDDING_SIZE=false

# Train
MAX_STEPS=500000
BATCH_SIZE=4

PROTEIN_SEQ_BATCH_SIZE=8
PROTEIN_GO_BATCH_SIZE=8
GO_GO_BATCH_SIZE=64
ACCUMULATION_STEPS=256
SCHEDULER_TYPE="linear"
WEIGHT_DECAY=0.01
OPTIMIZE_MEMORY=true

# MLM Loss
MLM_LAMBDA=1.0
MLM_LEARNING_RATE=1e-5
LM_WARMUP_RATIO=0.1

# KE Loss
KE_LAMBDA=1.0
KE_LEARNING_RATE=2e-5
KE_MAX_SCORE=12.0
KE_SCORE_FN='transE'
KE_WARMUP_RATIO=0.1


deepspeed --num_gpus=4 run_pretrain.py \
  --do_train \
  --output_dir $OUTPUT_DIR \
  --pretrain_data_dir $PRETRAIN_DATA_DIR \
  --protein_seq_data_file_name $PROTEIN_DATA_FILE_NAME \
  --in_memory $IN_MEMORY \
  --max_protein_seq_length $MAX_PROTEIN_SEQ_LENGTH \
  --model_protein_seq_data true \
  --model_protein_go_data true \
  --model_go_go_data true \
  --use_desc $USE_DESC \
  --max_text_seq_length $MAX_TEXT_SEQ_LENGTH \
  --dataloader_protein_go_num_workers $PROTEIN_GO_NUM_WORKERS \
  --dataloader_go_go_num_workers $GO_GO_NUM_WORKERS \
  --dataloader_protein_seq_num_workers $PROTEIN_SEQ_NUM_WORKERS \
  --num_protein_go_neg_sample $NUM_PROTEIN_GO_NEG_SAMPLE \
  --num_go_go_neg_sample $NUM_GO_GO_NEG_SAMPLE \
  --negative_sampling_fn $NEGTIVE_SAMPLING_FN \
  --protein_go_sample_head $PROTEIN_GO_SAMPLE_HEAD \
  --protein_go_sample_tail $PROTEIN_GO_SAMPLE_TAIL \
  --go_go_sample_head $GO_GO_SAMPLE_HEAD \
  --go_go_sample_tail $GO_GO_SAMPLE_TAIL \
  --protein_model_file_name $PROTEIN_MODEL_PATH \
  --text_model_file_name $TEXT_MODEL_PATH \
  --go_encoder_cls $GO_ENCODER_CLS \
  --protein_encoder_cls $PROTEIN_ENCODER_CLS \
  --ke_embedding_size $KE_EMBEDDING_SIZE \
  --double_entity_embedding_size $DOUBLE_ENTITY_EMBEDDING_SIZE \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --weight_decay $WEIGHT_DECAY \
  --optimize_memory $OPTIMIZE_MEMORY \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --lr_scheduler_type $SCHEDULER_TYPE \
  --mlm_lambda $MLM_LAMBDA \
  --lm_learning_rate $MLM_LEARNING_RATE \
  --lm_warmup_ratio $LM_WARMUP_RATIO \
  --ke_warmup_ratio $KE_WARMUP_RATIO \
  --ke_lambda $KE_LAMBDA \
  --ke_learning_rate $KE_LEARNING_RATE \
  --ke_max_score $KE_MAX_SCORE \
  --ke_score_fn $KE_SCORE_FN \
  --ke_warmup_ratio $KE_WARMUP_RATIO \
  --seed 2021 \
  --deepspeed dp_config.json \
  --fp16 \
  --dataloader_pin_memory \
