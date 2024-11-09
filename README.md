# UaIT
Can LLMs Learn Uncertainty on Their Own? Expressing Uncertainty Effectively in A Self-Training Manner (EMNLP 2024)

NOTE: This is the initial version code and will be updated and improved recently.

### Environments

Please config environment by following `requirements.txt`.

### Data Preparing
Training Dataset: https://drive.google.com/file/d/13z_qrVOBlgu75IJBpX-1vMSCC6hC9yH4/view?usp=sharing
Download and set the path in `parse_triviaqa_ft_chat.py`

```shell
cd src/parse_datasets
python parse_triviaqa_ft_chat.py
```

### Uncertainty Estimation

#### for the Trivia QA dataset:
```shell
cd src
sh scripts/trivia_qa/ue_pipeline_llama2-chat-7b.sh
```

### FT Data Construction
```shell
cd src/finetune
python get_ft_data.py --train-data-path [train-data-path] --train-data-auroc [train-data-auroc]
```

### Uncertainty-aware Instruction Tuning
```shell
cd src/finetune
sh train.sh 4 [data]
```

### Uncertainty Expression
```shell
cd src/finetune
sh scripts/trivia_qa/pe_pipeline_llama2-chat-7b.sh [lora_weights]
```

### Citation
```bibtex
@inproceedings{liu-etal-2024-llms-learn-uncertainty,
    title = "Can {LLM}s Learn Uncertainty on Their Own? Expressing Uncertainty Effectively in A Self-Training Manner",
    author = "Liu, Shudong  and
      Li, Zhaocong  and
      Liu, Xuebo  and
      Zhan, Runzhe  and
      Wong, Derek  and
      Chao, Lidia  and
      Zhang, Min",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1205",
    pages = "21635--21645",
}
```
