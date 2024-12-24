export CUDA_VISIBLE_DEVICES=4

model_name=SSCNN

  
python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_432_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 432 \
  --label_len 48 \
  --pred_len 336 \
  --cycle_len 144 \
  --short_period_len 12 \
  --kernel_size 4 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 8 \
  --spatial 0 \
  --short_term_attn 0 \
  --seasonal_attn 0 \
  --long_term_attn 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'