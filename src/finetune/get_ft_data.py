import json
import pickle
import tqdm
import numpy as np
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', type=str)
parser.add_argument('--train-data-auroc', type=str)
args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--non-chat', action='store_true', default=False)
args = parser.parse_args()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
verbalized_confidence = ['Uncertainly', 'Possibly', 'Moderately', 'Confidently', 'Absolutely']

with open(args.train_data_path,'r') as fr1:
    with open(args.train_data_auroc,'rb') as fr2:
        train_json = json.load(fr1)
        train_aoc = pickle.load(fr2)    
        normalized_confidence_score = [np.exp(-sc) for sc in train_aoc["uncertainty_score"]]

ft_data = []
correct_num = 0
incorrect_num = 0
skip_sample = []
chat_few_shot_prompt = 'Please directly return the answer to the following question without any explanation and indicate your level of confidence. Note that the confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. For instance, if your confidence level is 80%, it means you are 80.0% certain that your answer is correct and there is a 20.0% chance that it may be incorrect.\nQuestion: '
answer_prompt = '\nAnswer and Confidence (1-100): '
no_conf_answer_prompt = '\nAnswer: '
conf_answer_prompt = '\nConfidence (1-100) and Answer: '

for idx, train_sample in enumerate(tqdm.tqdm(train_json)):
    ft_data_sample = {}
    model_answer = train_aoc["predictions"][idx].split('nswer: ',1)[-1].split('is: ',1)[-1]
    golden_answer = train_sample["target"]
    if len(model_answer.split(' ')) > 6:
        skip_sample.append(idx)
        continue
    ft_data_sample["input"] = B_INST + chat_few_shot_prompt + train_sample["question"] + answer_prompt + E_INST
    confidence_score = round(100*normalized_confidence_score[idx],1)
    if train_aoc['correctness'][idx] == 1 and confidence_score>50:
        ft_data_sample["output"] = golden_answer + '; ' + str(confidence_score) + "%"
        correct_num += 1
        ft_data.append(ft_data_sample)
    elif train_aoc['correctness'][idx] == 0 and confidence_score<50:
        ft_data_sample["output"] = model_answer.rstrip('.').strip() + '; ' + str(confidence_score) + "%"
        incorrect_num += 1
        ft_data.append(ft_data_sample)

save_path = '../../src/finetune/dataset/tqa_llama2-chat/'
if os.path.exists('../../src/finetune/dataset'): 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Created")
    else:
        print("Exist")
else:
    print("ERROR")

with open(save_path+'train.json', "w") as out_json:
    json.dump(ft_data[:int(len(ft_data)*0.97)], out_json)
with open(save_path+'dev.json', "w") as out_json:
    json.dump(ft_data[int(len(ft_data)*0.97):], out_json)
with open(save_path+'dataset.info', "w") as out_info:
    out_info.write(str(len(skip_sample))+'\n')
    out_info.write(str(train_aoc['correctness'].sum())+' '+str(len(train_aoc['correctness']))+'\n')
    out_info.write(str(correct_num)+' '+str(incorrect_num)+'\n')
    out_info.write(str(int(len(ft_data)*0.97))+' '+str(len(ft_data)-int(len(ft_data)*0.97))+'\n')
