export CUDA_VISIBLE_DEVICES=4

model_name=SCNN

python -u our_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --label_len 48 \
  --pred_len 3 \
  --cycle_len 144 \
  --short_period_len 12 \
  --kernel_size 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 

python -u our_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --label_len 48 \
  --pred_len 24 \
  --cycle_len 144 \
  --short_period_len 12 \
  --kernel_size 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 

python -u our_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --label_len 48 \
  --pred_len 96 \
  --cycle_len 144 \
  --short_period_len 12 \
  --kernel_size 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 