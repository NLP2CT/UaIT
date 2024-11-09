import argparse
import os
import pickle
import random
import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import config

parser = argparse.ArgumentParser()
parser.add_argument('--generation-model', type=str, default='facebook/opt-350m')
parser.add_argument('--run-name', type=str, default='run_1')
parser.add_argument('--chat', action='store_true', default=False)
args = parser.parse_args()

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

# Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

run_name = args.run_name

tokenizer = AutoTokenizer.from_pretrained(args.generation_model, use_fast=False, cache_dir=config.data_dir)

with open(f'{config.output_dir}/{run_name}/raw_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)
    with open(f'{config.output_dir}/{run_name}/raw_generation.txt','w') as outfile:
        for sample in tqdm(sequences):
            cleaned_generations = torch.ones_like(sample['generations'])
            question = sample['question']
            generated_texts = sample['generated_texts']
            few_shot_question = sample['few_shot_question']
            cleaned_generated_texts = []
            # print(cleaned_generations)
            # outfile.write(cleaned_generations)
            outfile.write("[few_shot_QUESTION]: "+few_shot_question+'\n')
            outfile.write("[Most Likely Gen]: "+sample['most_likely_generation']+'\n')
            outfile.write("[Generated Texts]: "+str(generated_texts)+'\n')

cleaned_sequences = []

for sample in tqdm(sequences):
    cleaned_generations = torch.ones_like(sample['generations'])
    question = sample['question']
    generated_texts = sample['generated_texts']
    cleaned_generated_texts = []

    max_len_of_generations = cleaned_generations.shape[-1]

    strings_to_filter_on = [
        ';', '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]

    for i, generated_text in enumerate(generated_texts):
        # if args.chat:
        #     generated_text = generated_text.lstrip().split('Answer: ',1)[-1]
        # else:
        #     for string in strings_to_filter_on:
        #         if string in generated_text:
        #             generated_text = generated_text.split(string)[0]
        cleaned_generated_texts.append(generated_text)
        # print(generated_text,tokenizer(generated_text)['input_ids'][1:])
        clean_ids = torch.cat(
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])
        # print(clean_ids,clean_ids[:max_len_of_generations],min(len(clean_ids), max_len_of_generations))
        cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]
        # print("cleaned_generations: ",cleaned_generations)
    sample['cleaned_generated_texts'] = cleaned_generated_texts
    sample['cleaned_generations'] = cleaned_generations

    cleaned_sequences.append(sample)

with open(f'{config.output_dir}/{run_name}/generations.pkl', 'wb') as outfile:
    pickle.dump(cleaned_sequences, outfile)

with open(f'{config.output_dir}/{run_name}/generations.txt','w') as outfile:
    for sample in tqdm(cleaned_sequences):
        cleaned_generations = torch.ones_like(sample['generations'])
        question = sample['question']
        generated_texts = sample['cleaned_generated_texts']
        cleaned_generated_texts = []
        # print(cleaned_generations)
        # outfile.write(cleaned_generations)
        outfile.write(question)
        outfile.write(str(generated_texts))
        outfile.write('\n')