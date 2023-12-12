export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_3 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_24 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'