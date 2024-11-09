
import string
import argparse
import pathlib
import pickle
import random
import accelerate
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.chat_utils import format_tokens,read_dialogs_from_file
import json
import config
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
# 'meta-llama/Llama-2-7b-chat-hf' 'mistralai/Mistral-7B-Instruct-v0.1'
parser.add_argument('--prompt-uncertainty', action='store_true', default=True)
parser.add_argument('--zero-shot', action='store_true', default=False)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
seed_value = 10

if 'opt' in args.model:
    model_name = 'opt'
elif 'Llama-2' and 'chat' in args.model:
    model_name = 'llama2-chat'
elif 'Llama-2' in args.model:
    model_name = 'llama2'
elif 'Mistral' and 'Instruct' in args.model:
    model_name = 'mistral-chat'
elif 'Mistral' in args.model:
    model_name = 'mistral'
elif 'llama' in args.model:
    model_name = 'llama'
elif 'gpt' in args.model:
    model_name = 'gpt'
elif 'falcon' in args.model:
    model_name = 'falcon'
else:
    raise NotImplementedError
if args.prompt_uncertainty:
    model_name += '_prompt_uncertainty'
if args.zero_shot:
    model_name += '_0shot'

if not pathlib.Path(f'{config.data_dir}/trivia_qa/{model_name}').exists():

    print('Preprocessing dataset')
    val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="validation")
    # train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")

    if args.prompt_uncertainty:
        few_shot_prompt = 'Here are some examples.\nQuestion: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nAnswer and Confidence (0-100): Sinclair Lewis; 90%\nQuestion: Where in England was Dame Judi Dench born?\nAnswer and Confidence (0-100): York; 70%\nQuestion: In which decade did Billboard magazine first publish and American hit chart?\nAnswer and Confidence (0-100): 30s; 80%\n\n'+'According to the format of the above examples, directly write the answer with one or few words to the following question without any explanation and indicate your level of confidence. Note that the confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. For instance, if your confidence level is 80%, it means you are 80% certain that your answer is correct and there is a 20% chance that it may be incorrect.\nQuestion: '
        instruct_prompt = 'Please directly return the answer to the following question without any explanation and indicate your level of confidence. Note that the confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. For instance, if your confidence level is 80.0%, it means you are 80.0% certain that your answer is correct and there is a 20.0% chance that it may be incorrect.\nQuestion: '
        answer_prompt = "\nAnswer and Confidence (0-100): "

    else:
        few_shot_prompt = 'Here are some examples.\nQuestion: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nAnswer: Sinclair Lewis\nQuestion: Where in England was Dame Judi Dench born?\nAnswer: York\nQuestion: In which decade did Billboard magazine first publish and American hit chart?\nAnswer: 30s\n\n'+'According to the format of the above examples, directly write the answer with one or few words to the following question without any explanation.\nQuestion: '
        instruct_prompt = 'Directly write the short answer to the following question without any explanation or introduction.\nQuestion: '
        # few_shot_prompt = 'Here is a reference:\n'
        # few_shot_prompt += 'Write a short answer directly without any explanation and introduction:\n'
        answer_prompt = "\nAnswer: "

    chat_data = {}

    for idx, sample in enumerate(tqdm.tqdm(val_data)):
        if args.zero_shot:
            chat_data_sample = [{"role":"user", "content":instruct_prompt+sample["question"]+answer_prompt},{"role":"assistant", "content":sample["answer"]["value"]}]
        else:
            chat_data_sample = [{"role":"user", "content":few_shot_prompt+sample["question"]+answer_prompt},{"role":"assistant", "content":sample["answer"]["value"]}]
        chat_data[sample["question_id"]] = chat_data_sample
    with open(f'{config.data_dir}/chat_json/trivia_qa/{model_name}_val.json', "w")as out_json:
        json.dump(chat_data, out_json)
    

    batch_size = 256  # change to 16 for full training

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        with open(f'{config.data_dir}/chat_json/trivia_qa/{model_name}_val.json', "r") as input_json:
            dialogs = json.load(input_json)
            dialogs_input = [dialogs[q_id][:-1] for q_id in batch["question_id"]]
            dialogs_input = format_tokens(dialogs_input,tokenizer=tokenizer)
            dialogs_output = [dialogs[q_id][-1] for q_id in batch["question_id"]]
            dialogs_output = [sample_output['content'] for sample_output in dialogs_output]
        answers = dialogs_output
        inputs = tokenizer(dialogs_input, padding=False, truncation=False)
        outputs = tokenizer(dialogs_output, padding=False, truncation=False)
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch['answer'] = answers
        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)
    val_data.save_to_disk(f'{config.data_dir}/trivia_qa/{model_name}')
    with open(f'{config.data_dir}/trivia_qa/{model_name}'+'/val.json','w') as out_json:
        json.dump(chat_data, out_json)
else:

    val_data = datasets.load_from_disk(f'{config.data_dir}/trivia_qa/{model_name}')
