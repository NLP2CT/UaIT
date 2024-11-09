target_model_name="mistralai/Mistral-7B-Instruct-v0.1"

dataset_name="sciq"
fraction_of_data=1.0
prompt_uncertainty=''
zero_shot=''
special_modelname='' #'scale10'
num_of_gens=5
relevance_model_name="cross-encoder/stsb-roberta-large"
model_generation_temperature=0.5
max_length_of_generation=64
decoding_method="greedy"
num_beams=1
devices=0
data_split="val"

run_name=${target_model_name}/${dataset_name}${fraction_of_data}$prompt_uncertainty/numbeams-${num_beams}/max_len_of_gen-${max_length_of_generation}/${num_of_gens}gens${special_modelname}

CUDA_VISIBLE_DEVICES=${devices} python generate-chat.py --num-generations-per-prompt ${num_of_gens} --dataset ${dataset_name} \
--fraction-of-data-to-use ${fraction_of_data} --model ${target_model_name} --temperature ${model_generation_temperature} \
--run-name ${run_name} --max-length-of-generation ${max_length_of_generation} --num-beams ${num_beams} \
--decoding-method ${decoding_method}  --data-split ${data_split}

CUDA_VISIBLE_DEVICES=${devices} python clean_generated_strings.py --generation-model ${target_model_name} --run-name ${run_name} --chat

CUDA_VISIBLE_DEVICES=${devices} python get_semantic_clusters.py --generation-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python get_likelihoods.py --evaluation-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python get_tokenwise_importance.py --measurement-model ${relevance_model_name} \
--tokenizer-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python get_sentence_similarities.py --measurement-model ${relevance_model_name} \
--run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python compute_uncertainty.py --senten-sim-meas-model ${relevance_model_name} \
--token-impt-meas-model ${relevance_model_name} --run-name ${run_name} --methods sar --metrics rougeL_to_target --threshold 0.3 --num-generation 1
