cd ../

# Training on 1 v100
nohup sh run_main.sh \
      --model ./model/remote_homology/ProtBertModel \
      --output_file remote_homology-ProtBert \
      --task_name remote_homology \
      --do_train False \
      --epoch 20 \
      --mean_output True \
      --optimizer AdamW \
      --per_device_batch_size 8 \
      --gradient_accumulation_steps 8 \
      --eval_step 50 \
      --eval_batchsize 8 \
      --warmup_ratio 0.08 \
      --frozen_bert False >./task/result/remote_homology-prediction.out 2>&1
