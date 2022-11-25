cd ../

# Training on 2 v100s
nohup sh run_main.sh \
      --model ./model/ss3/ProtBertModel \
      --output_file ss3-ProtBert \
      --task_name ss3 \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --eval_step 100 \
      --eval_batchsize 4 \
      --warmup_ratio 0.08 \
      --frozen_bert False >./task/result/ss3-ProtBert.out 2>&1
