cd ../

# Training on 4 v100 GPUs
nohup sh run_main.sh \
      --model ./model/contact/OntoproteinModel \
      --output_file contact-ontoprotein \
      --task_name contact \
      --do_train False \
      --epoch 10 \
      --optimizer AdamW \
      --per_device_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --eval_step 50 \
      --eval_batchsize 1 \
      --warmup_ratio 0.08 \
      --frozen_bert True >./task/result/contact-ontoprotein.out 2>&1
