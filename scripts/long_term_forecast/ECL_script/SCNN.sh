export CUDA_VISIBLE_DEVICES=2

model_name=SCNN

python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 1 \
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
  --model_id ECL_168_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --pred_len 96 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 1 \
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
  --model_id ECL_336_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 1 \
  --learning_rate 0.0005 \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --lradj 'type3'