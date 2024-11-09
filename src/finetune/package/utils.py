import os
import sys
from transformers import HfArgumentParser, TrainingArguments
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from .hparams import ModelArguments, DataArguments


_TRAIN_ARGS = [
    ModelArguments, DataArguments, TrainingArguments
]
_TRAIN_CLS = Tuple[
    ModelArguments, DataArguments, TrainingArguments
]

def parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    else:
        return parser.parse_args_into_dataclasses()

def parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return parse_args(parser, args)

def init_training_args(args):
    args.auto_find_batch_size = True
    args.optim = "paged_adamw_32bit" if args.bf16 else "adamw_torch" #"paged_adamw_32bit"
    args.save_strategy = "epoch"
    if args.fp16 == False:
        args.bf16 = True