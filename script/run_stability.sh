cd ../

# Training on 1 v100
nohup sh run_main.sh \
      --model ./model/ontoprotein \
      --output_file stability-ontoprotein \
      --task_name stability \
      --do_train True \
      --epoch 5 \
      --mean_output True \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 16 \
      --eval_step 50 \
      --eval_batchsize 16 \
      --warmup_ratio 0.08 \
      --frozen_bert False >./task/result/stability-ontoprotein.out 2>&1
