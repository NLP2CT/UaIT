import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default = None,
        )
    quant: Optional[str] = field(
        default = None,
        )
    using_lora: bool = field(
        default=False,
        )
    lora_alpha: int = field(
        default = 16,
        )
    lora_r: int = field(
        default = 64,
        )
    lora_dropout: float = field(
        default = 0.05,
        )
    use_fast_tokenizer: bool = field(
        default = False,
        )
    use_flash_attention_2: bool = field(
        default=False,
        )
    left_pad: bool = field(
        default=False,
        )
    response_template: str = field(
        default=None,
        )


@dataclass
class DataArguments:
    train_data_file: Optional[str] = field(
        default=".",
        )
    eval_data_file: Optional[str] = field(
        default = None)
    max_seq_length: int = field(
        default=1024,
        )
    ctx_trainer: bool = field(default=False)
    


