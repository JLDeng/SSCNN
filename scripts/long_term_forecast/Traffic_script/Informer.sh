export CUDA_VISIBLE_DEVICES=4

model_name=Informer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_168_3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_168_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_168_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'