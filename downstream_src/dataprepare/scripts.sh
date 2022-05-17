python ./healthcaremagic.py ./raw_data/healthcaremagic/

python ./icliniq.py ./raw_data/icliniq/icliniq_dialogue.txt

python ./mediqa-mas.py ./raw_data/mediqa-mas/mediqa-ans/question_driven_answer_summarization_primary_dataset.json ./raw_data/mediqa-mas/test_task2 ./raw_data/mediqa-mas/val_task2

python ./mediqa-qs.py ./raw_data/mediqa-qs/raw_train.xlsx ./raw_data/mediqa-qs/raw_test.xlsx ./raw_data/mediqa-qs/raw_val.xlsx

python ./mediqa-ans.py ./raw_data/mediqa-mas/mediqa-ans/question_driven_answer_summarization_primary_dataset.json ./qdriven-chiqa-summarization/data_processing/data/medinfo_section2answer_validation_data_with_question.json ./qdriven-chiqa-summarization/data_processing/data/bioasq_abs2summ_training_data_with_question.json







