cd ../

nohup sh run_main.sh \
      --model ./model/transformers/ProtBert \
      --output_file fluorescence-ProtBert \
      --task_name fluorescence \
      --do_train True \
      --epoch 15 \
      --mean_output True \
      --optimizer Adam \
      --per_device_batch_size 4 \
      --gradient_accumulation_steps 16 \
      --eval_step 50 \
      --eval_batchsize 32 \
      --warmup_ratio 0.0 \
      --frozen_bert True >./task/result/fluorescence-ProtBertModel.out 2>&1
