from collections import defaultdict, OrderedDict
from typing import Dict, Optional


CHOICES = ["A", "B", "C", "D"]

DEFAULT_MODULE = defaultdict(str)

DEFAULT_TEMPLATE = defaultdict(str)

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LOG_FILE_NAME = "trainer_log.jsonl"

METHODS = ["full", "freeze", "lora"]

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "Pre-Training": "pt"
}


def register_model_group(
    models: Dict[str, str],
    module: Optional[str] = None,
    template: Optional[str] = None
) -> None:
    prefix = None
    for name, path in models.items():
        if prefix is None:
            prefix = name.split("-")[0]
        else:
            assert prefix == name.split("-")[0], "prefix should be identical."
        SUPPORTED_MODELS[name] = path
    if module is not None:
        DEFAULT_MODULE[prefix] = module
    if template is not None:
        DEFAULT_TEMPLATE[prefix] = template

