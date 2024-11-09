from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
import torch
from peft import LoraConfig, get_peft_model
from package.utils import parse_train_args, init_training_args
from package.plotting import plot_loss
from package.callbacks import LogCallback
from typing import Any, Dict, Optional, Tuple
import numpy as np
import random
from trl.trainer.utils import PeftSavingCallback
from package.ctx_trainer import CTXTrainer
from package.DataCollatorForMixContext import DataCollatorForMixContext


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_data(args):
    if args.eval_data_file is not None:
        data_files = {
            "train":args.train_data_file,
            "eval":args.eval_data_file,
        }
    else:
        data_files = {
            "train":args.train_data_file,
        }
    dataset = load_dataset(
        "json",
        data_files = data_files,
        )
    train_dataset = dataset["train"]
    if args.eval_data_file is not None:
        eval_dataset = dataset["eval"]
    else:
        eval_dataset = None
    
    return train_dataset, eval_dataset

def load_lora(args):
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_r = args.lora_r

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    return peft_config

def load_model(model_args, training_args):
    model_name = model_args.model_name
    if model_args.use_flash_attention_2:
        training_args.fp16 = False
        training_args.bf16 = True
    # quant
    if model_args.quant is not None:
        if model_args.quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = torch.bfloat16 if training_args.bf16 else None,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit = True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            torch_dtype = torch.bfloat16 if training_args.bf16 else None,
            use_flash_attention_2 = model_args.use_flash_attention_2,
        )
        print(bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.bfloat16 if training_args.bf16 else None,
            use_flash_attention_2 = model_args.use_flash_attention_2,
        )
    model.config.use_cache = False

    return model






def exc(args: Optional[Dict[str, Any]] = None):

    
    # args
    model_args, data_args, training_args = parse_train_args(args)

    # model

    ## backbone
    model = load_model(model_args, training_args)

    ## lora
    if model_args.using_lora:
        peft_config = load_lora(model_args)
        # model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    ## tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        use_fast = model_args.use_fast_tokenizer,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token#"[PAD]"
    tokenizer.padding_side = "right"
    
    print("padding_side:{}".format(tokenizer.padding_side))

    # dataset

    ## load data
    train_dataset, eval_dataset = load_data(data_args)
    max_seq_length = data_args.max_seq_length

    ## collator
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            text = f"{example['input'][i]} {example['output'][i]}</s>"
            output_texts.append(text)
        return output_texts
    
    # def ctx_formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['input'])):
    #         context = exa


    if model_args.response_template == None:
        response_template = "\n### Response:"
    else:
        # response_template = "\n{}".format(model_args.response_template)
        response_template = "{}".format(model_args.response_template)
        # response_template = "(0-100):"
    CollatorType = DataCollatorForMixContext if data_args.ctx_trainer else DataCollatorForCompletionOnlyLM
    collator = CollatorType(response_template, tokenizer=tokenizer)

    # training args

    init_training_args(training_args)
    setup_seed(training_args.seed)

    # trainer
    callbacks = [LogCallback(), PeftSavingCallback()]

    TrainerType = CTXTrainer if data_args.ctx_trainer else SFTTrainer
    
    if model_args.using_lora:
        trainer = TrainerType(
            model,
            train_dataset = train_dataset,
            formatting_func = formatting_prompts_func,
            data_collator = collator,
            eval_dataset = eval_dataset,
            peft_config = peft_config,
            max_seq_length = max_seq_length,
            tokenizer = tokenizer,
            args = training_args,
            callbacks = callbacks,
        )
    else:
        trainer = TrainerType(
            model,
            train_dataset = train_dataset,
            formatting_func = formatting_prompts_func,
            data_collator = collator,
            eval_dataset = eval_dataset,
            max_seq_length = max_seq_length,
            tokenizer = tokenizer,
            args = training_args,
            callbacks = callbacks,
        )
    print(trainer.args)
    print(trainer.callback_handler.callbacks)

    # training
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])



if __name__ == "__main__":
    exc()
