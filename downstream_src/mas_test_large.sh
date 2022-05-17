export CUDA_VISIBLE_DEVICES=$1
python summarization.py \
    --model_name_or_path $2 \
    --train_file ./dataprepare/data/mediqa-mas/train.json \
    --validation_file ./dataprepare/data/mediqa-mas/dev.json \
    --test_file ./dataprepare/data/mediqa-mas/test.json \
    --text_column src \
    --summary_column tgt \
    --source_prefix " " \
    --num_beams 5 \
    --val_max_target_length 128 \
    --val_min_target_length 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 6 \
    --learning_rate 1e-5 \
    --output_dir /platform_tech/yuanzheng/biobart_downstream/mediqa-mas/$3 \
    --testing_dir /platform_tech/yuanzheng/biobart_downstream/mediqa-mas/11wl/3/ \
    --testing_dir_contrast /platform_tech/yuanzheng/biobart_downstream/mediqa-mas/lar/2/ \
    --only_test \