MODEL_PATH=$1
DATA_PATH=$2
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
          ./train.py --config-file ./bart.json \
                     --output_dir $MODEL_PATH \
                     --token_nosing_prob 0.1 \
                     --max_seq_length 512 \
                     --max_predictions_per_seq 150 \
                     --seed 42 \
                     --lr_schedule LL \
                     --job_name biobart_pretrain \
                     --print_steps 10 \
                     --save_steps 100 \
                     --data_path_prefix $DATA_PATH \
          --deepspeed --deepspeed_config ./ds_config_zero2.json
