export CUDA_VISIBLE_DEVICES=1

model_name=SSCNN


python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 96 \
  --long_term 1 \
  --seasonal 1 \
  --seasonal_attn 1 \
  --cycle_len 24 \
  --short_term 1 \
  --short_term_attn 1 \
  --short_period_len 8 \
  --spatial 1 \
  --kernel_size 2 \
  --e_layers 4 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'


python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 192 \
  --long_term 1 \
  --seasonal 1 \
  --seasonal_attn 1 \
  --cycle_len 24 \
  --short_term 1 \
  --short_term_attn 1 \
  --short_period_len 8 \
  --spatial 1 \
  --kernel_size 2 \
  --e_layers 4 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'

python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --long_term 1 \
  --seasonal 1 \
  --seasonal_attn 1 \
  --cycle_len 24 \
  --short_term 1 \
  --short_term_attn 1 \
  --short_period_len 8 \
  --spatial 1 \
  --kernel_size 2 \
  --e_layers 4 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'
