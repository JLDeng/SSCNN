export CUDA_VISIBLE_DEVICES=4

model_name=Normformer

python -u our_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_168_24 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5