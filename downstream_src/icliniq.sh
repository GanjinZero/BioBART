# ./icliniq_script.sh 6 /platform_tech/yuanzheng/biobart_ckps/biobart_base_0213_conti_20w/saved_models/biobart_pretrain/checkpoint_global_step_72332,epoch_index_1001/

# ./icliniq_script.sh 6 /platform_tech/yuanzheng/biobart_ckps/biobart_base_0213_conti_20w/saved_models/biobart_pretrain/checkpoint_global_step_122894,epoch_index_1701/

# ./icliniq_script.sh 6 /platform_tech/yuanzheng/biobart_ckps/biobart_base_0213_conti_20w/saved_models/biobart_pretrain/checkpoint_global_step_43483,epoch_index_601/

# ./icliniq_script.sh 6 /platform_tech/yuanzheng/biobart_ckps/biobart_base_512_cased_ver1028/saved_models/biobart_pretrain/checkpoint_global_step_39111,epoch_index_541

# ./icliniq_script.sh 6 /platform_tech/yuanzheng/biobart_ckps/biobart_base_512_cased_ver1103/saved_models/biobart_pretrain/checkpoint_global_step_39156,epoch_index_541


# ./icliniq_script.sh 6 ../entity_linking_t5/bart-base

# ./icliniq_script_large.sh 7 ../entity_linking_t5/bart-large


model=/platform_tech/yuanzheng/biobart_ckps/biobart-large/biobart_wo_permute_large/saved_models/biobart_pretrain/checkpoint_global_step_108544,epoch_index_1501/

./icliniq_script_large.sh 7 $model 11wl

./mediqa-mas_script_large.sh 7 $model 11wl

./mediqa-ans_script_large.sh 7 $model 11wl

./mediqa-qs_script_large.sh 7 $model 11wl

./covid-dialogue_script_large.sh 7 /platform_tech/yuanzheng/biobart_ckps/biobart-large/biobart_wo_permute_large/saved_models/biobart_pretrain/checkpoint_global_step_108544,epoch_index_1501/ 11wl

model=../entity_linking_t5/bart-large

./icliniq_script_large.sh 7 $model lar

./mediqa-mas_script_large.sh 7 $model lar

./mediqa-ans_script_large.sh 7 $model lar

./mediqa-qs_script_large.sh 7 $model lar

./covid-dialogue_script_large.sh 7 $model lar
