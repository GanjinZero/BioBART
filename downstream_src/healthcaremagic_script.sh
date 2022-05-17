export CUDA_VISIBLE_DEVICES=$1
python summarization.py \
    --model_name_or_path $2 \
    --train_file ./dataprepare/data/healthcaremagic/train.json \
    --validation_file ./dataprepare/data/healthcaremagic/dev.json \
    --test_file ./dataprepare/data/healthcaremagic/test.json \
    --text_column src \
    --summary_column tgt \
    --source_prefix " " \
    --num_beams 5 \
    --val_max_target_length 30 \
    --val_min_target_length 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 6 \
    --output_dir /platform_tech/yuanzheng/biobart_downstream/healthcaremagic/