#model_args
model_size=7b
model_style=-chat
model_name="mistralai/Mistral-7B-Instruct-v0.1"
lora_alpha=16
lora_r=64
lora_dropout=0.05
response_template="[/INST]"
model_args="--response_template $response_template --model_name $model_name --using_lora --lora_alpha $lora_alpha --lora_r $lora_r --lora_dropout $lora_dropout --use_fast_tokenizer"
model_output_dir=${model_name}-using_lora-lora_alpha-$lora_alpha-lora_r-$lora_r-lora_dropout-$lora_dropout

#data_args
data_dir=$2 #sciq_mistral-chat
train_data_file=dataset/$data_dir/train.json
eval_data_file=dataset/$data_dir/dev.json
max_seq_length=1024
data_args="--train_data_file $train_data_file --eval_data_file $eval_data_file --max_seq_length $max_seq_length"
data_output_dir=data-$data_dir-max_seq_len-$max_seq_length

#training_args
seed=42
data_seed=$seed
num_train_epochs=$1
per_device_train_batch_size=2
per_device_eval_batch_size=32
gradient_accumulation_steps=16
learning_rate=2e-4
lr_scheduler_type="cosine"
warmup_ratio=0.03
training_output_dir=seed-$seed-epochs-$num_train_epochs-bsz-$per_device_train_batch_size-freq-$gradient_accumulation_steps-lr-$learning_rate-scheduler-$lr_scheduler_type-warmup_ratio-$warmup_ratio
#logging_eval_args
save_total_limit=2
logging_strategy="steps"
logging_steps=50
evaluation_strategy="steps"
prediction_loss_only=True
eval_steps=100

logging_eval_args="--logging_strategy $logging_strategy --logging_steps $logging_steps --evaluation_strategy $evaluation_strategy --prediction_loss_only --eval_steps $eval_steps --save_total_limit $save_total_limit --seed $seed --data_seed $data_seed"

final_output_dir=./results/$model_output_dir/$data_output_dir/$training_output_dir

mkdir -p $final_output_dir

training_args="--output_dir $final_output_dir --num_train_epochs $num_train_epochs --per_device_train_batch_size $per_device_train_batch_size --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --lr_scheduler_type $lr_scheduler_type --warmup_ratio $warmup_ratio --fp16 --do_train --do_eval --per_device_eval_batch_size $per_device_eval_batch_size"


python sft.py \
    $model_args \
    $data_args \
    $training_args \
    $logging_eval_args \

