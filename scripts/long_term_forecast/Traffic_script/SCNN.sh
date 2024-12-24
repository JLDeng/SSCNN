export CUDA_VISIBLE_DEVICES=7

model_name=SCNN

python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'