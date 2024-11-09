import argparse
import pathlib
import pickle
import sys

import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
import tqdm
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--type-of-question', type=str)
parser.add_argument('--num-generations-per-prompt', type=int, default=5)
parser.add_argument('--fraction-of-data-to-use', type=float, default=1.0)
parser.add_argument('--model', type=str, default='facebook/opt-350m')
parser.add_argument('--run-name', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--num-beams', type=int, default=5)
parser.add_argument('--decoding-method', type=str, default='beam_search')
parser.add_argument('--top-p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--most-likely-gen-beams', type=int, default=5)
parser.add_argument('--prompt-uncertainty', action='store_true', default=False)
parser.add_argument('--zero-shot', action='store_true', default=False)
parser.add_argument('--data-split', type=str, default='val')
parser.add_argument("--lora-weights", type=str, default=None, help="use lora")
parser.add_argument('--special-modelname', type=str, default=None)
args = parser.parse_args()

run_name = args.run_name

device = 'cuda:0'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
np.random.seed(seed_value)

# Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

if args.model == 'facebook/opt-30b':
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True).cuda()

if args.lora_weights is not None and os.path.exists(args.lora_weights):
    print(f'Loading LoRA weights from path: {args.lora_weights}')
    model = PeftModel.from_pretrained(model, args.lora_weights, torch_dtype=torch.float16)
# print(model.hf_device_map)

# print(model)
# sys.exit()


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
if args.data_split=='train':
    model_name += '_ft'
if args.special_modelname != None:
    model_name += '_'+args.special_modelname


if args.dataset == 'coqa':
    dataset = datasets.load_from_disk(f'{config.data_dir}/coqa_dataset')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif args.dataset == 'trivia_qa':
    print(f'{config.data_dir}/trivia_qa/{model_name}')
    dataset = datasets.load_from_disk(f'{config.data_dir}/trivia_qa/{model_name}')
elif args.dataset == 'natural_questions':
    dataset = datasets.load_from_disk(f'{config.data_dir}/natural_questions_{model_name}')
elif args.dataset == 'sciq':
    print(f'{config.data_dir}/sciq/{model_name}')
    dataset = datasets.load_from_disk(f'{config.data_dir}/sciq/{model_name}')
elif args.dataset == 'medqa':
    print(f'{config.data_dir}/medqa/{model_name}')
    dataset = datasets.load_from_disk(f'{config.data_dir}/medqa/{model_name}')
else:
    raise NotImplementedError

non_coqa_list = ['trivia_qa', 'natural_questions', 'sciq', 'medqa', 'tqa_retrieval']

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset


def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset


if args.dataset == 'coqa':
    questions = encode_and_format_dataset(train_dataset)
elif args.dataset in non_coqa_list:
    questions = train_dataset

dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")
bertscore_metric = evaluate.load('bertscore')

def extract_string(prediction,reference):

    if len(prediction.split()) >= 6:
        if "nswer:" in prediction:
            return prediction.split("nswer:", 1)[1]
        elif prediction.count('"') == 2:
            start_index = prediction.index('"') + 1
            end_index = prediction.index('"', start_index)
            return prediction[start_index:end_index]
        elif "question:" in prediction:
            return prediction.split("question:", 1)[1]
        elif "is" in prediction or "was" in prediction or "are" in prediction or "were" in prediction:
            keywords = ["\n","is", "was", "are", "were"]
            for keyword in keywords:
                if keyword in prediction:
                    pre_key = rouge.compute(predictions=[prediction.split(keyword, 1)[0]],references=[reference])['rougeL']
                    post_key = rouge.compute(predictions=[prediction.split(keyword, 1)[1]],references=[reference])['rougeL']
                    if post_key > pre_key:
                        return prediction.split(keyword, 1)[1]
                    else:
                        return prediction.split(keyword, 1)[0]
                
    return prediction


def get_most_likely_generation(input_ids, args, max_length_of_generated_sequence, num_beam, num_return_sequences):
    if args.decoding_method == 'beam_search':
        most_likely_generation = model.generate(input_ids,
                                                num_beams=num_beam, # num_beam=5, beam search, 
                                                num_return_sequences=num_return_sequences, # num_return_sequences=2
                                                do_sample=False,
                                                max_length=input_ids.shape[1] +
                                                           max_length_of_generated_sequence)
    elif args.decoding_method == 'greedy':
        most_likely_generation = model.generate(inputs=input_ids,
                                                num_beams=1,
                                                do_sample=False,
                                                max_length=input_ids.shape[1] +
                                                           max_length_of_generated_sequence,
                                                temperature = args.temperature,
                                                use_cache = True,
                                                repetition_penalty = 1.0,
                                                length_penalty = 1,
                                                top_p = args.top_p,
                                                )
    else:
        raise NotImplementedError

    return most_likely_generation


def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = args.max_length_of_generation
        sequences = []
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            if args.dataset in non_coqa_list:
                input_ids = batch['input_ids'].to(device).reshape(1, -1)
            else:
                input_ids = batch['input_ids'].to(device)

            most_likely_generation = get_most_likely_generation(input_ids, args, max_length_of_generated_sequence,
                                                                num_beam=5, num_return_sequences=2)
            input_length = input_ids.shape[1] if args.dataset in non_coqa_list else \
                batch['input_ids'].shape[1]

            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=device)
            for i in range(number_of_generations):
                generation = model.generate(inputs=input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                            temperature=args.temperature,
                                            use_cache = True,
                                            repetition_penalty = 1.0,
                                            length_penalty = 1,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                    
                elif args.dataset in non_coqa_list:
                    few_shot_question = tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': tokenizer.decode(input_ids[0]),
                        'question': question
                    }


                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[input_length:], skip_special_tokens=True))

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][input_length:], skip_special_tokens=True)
                if args.decoding_method == 'beam_search':
                    sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                    sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                        most_likely_generation[1][input_length:], skip_special_tokens=True)

                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]
                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0

                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']

                if args.dataset == 'coqa':
                    sequence_dict['additional_answers'] = [x[0] for x in batch['additional_answers']]
                elif args.dataset in non_coqa_list : 
                    sequence_dict['additional_answers'] = None
                elif args.dataset == 'natural_questions':
                    sequence_dict['additional_answers'] = [x for x in batch['additional_answers']]
                else:
                    raise NotImplementedError

                sequence_dict['exact_match'] = 0.0

                sequence_dict['bertscore_precision'] = 0.0
                sequence_dict['bertscore_recall'] = 0.0
                sequence_dict['bertscore_f1'] = 0.0

                if args.dataset == 'coqa':
                    reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']]
                elif args.dataset == 'natural_questions':
                    reference_answers = batch['answer'] + [x[0] for x in batch['additional_answers']]
                elif args.dataset in non_coqa_list:  
                    reference_answers = batch['answer']

                for answer in reference_answers:
                    if 'llama' or 'Mistral' in args.model:
                        # clean most likely LLaMA generations
                        if args.prompt_uncertainty:
                            # predictions = [sequence_dict['most_likely_generation'].lstrip().split('onfidence: ',1)[-1]]
                            if not args.zero_shot:
                                if 'nswer:' in sequence_dict['most_likely_generation']:
                                    predictions = [sequence_dict['most_likely_generation'].lstrip().split(';',1)[0].split('nswer:',1)[-1].split('\n',1)[0]]
                                else:
                                    predictions = [sequence_dict['most_likely_generation'].lstrip().split(';',1)[0].split(':',1)[-1].split('\n',1)[0]]
                                # predictions = [extract_string(predictions,answer).strip()]
                            else:
                                # predictions = [sequence_dict['most_likely_generation'].lstrip().split(';',1)[0].split(':',1)[-1].strip()]
                                predictions = [sequence_dict['most_likely_generation'].lstrip().split(';',1)[0].lstrip().split('onfidence',1)[0].split('nswer:',1)[-1].strip()]
                                # predictions = [sequence_dict['most_likely_generation'].lstrip().split(';',1)[-1].lstrip().split('nswer',1)[-1].strip()]
                                # predictions = [extract_string(predictions,answer).strip()]

                        else:
                            # predictions = [sequence_dict['most_likely_generation'].lstrip().split('\n')[0]]
                            # predictions = [sequence_dict['most_likely_generation'].lstrip().split('Answer: ',1)[-1]]
                            # predictions = [sequence_dict['most_likely_generation'].split('nswer:',1)[-1].strip() if 'nswer:' in sequence_dict['most_likely_generation'] else sequence_dict['most_likely_generation'].split(':')[-1].strip()]
                            predictions = sequence_dict['most_likely_generation']
                            predictions = [extract_string(predictions,answer).strip()]
                    else:
                        predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    bertscore_results = bertscore_metric.compute(predictions=predictions, references=references,
                                                                 lang="en")
                    sequence_dict['bertscore_precision'] = max(bertscore_results['precision'][0],
                                                               sequence_dict['bertscore_precision'])
                    sequence_dict['bertscore_recall'] = max(bertscore_results['recall'][0],
                                                            sequence_dict['bertscore_recall'])
                    sequence_dict['bertscore_f1'] = max(bertscore_results['f1'][0], sequence_dict['bertscore_f1'])

                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type],
                                                                       sequence_dict[rouge_type + '_to_target'])
                sequences.append(sequence_dict)

    return sequences


sequences = get_generations(model, dataloader, args.num_generations_per_prompt)

pathlib.Path(f'{config.output_dir}/' + run_name).mkdir(parents=True, exist_ok=True)

with open(f'{config.output_dir}/{run_name}/raw_generations.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)