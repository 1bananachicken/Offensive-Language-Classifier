import torch
import pandas as pd
from datasets import Dataset
from modelscope import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer


def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个网络攻击性语言辨别程序，对于每一个输入，如果具有冒犯性，就输出1，否则输出0"
                            f"<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
                            f"<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


df_t = pd.read_json('./proceed_dataset/homo/train.json')
df_v = pd.read_json('./proceed_dataset/homo/val.json')
train_ds = Dataset.from_pandas(df_t)
val_ds = Dataset.from_pandas(df_v)
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./weights', revision='master')
# model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir='./weights', revision='master')

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:1",torch_dtype=torch.bfloat16)
tokenizer.padding_side = 'right'




train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
val_dataset = val_ds.map(process_func, remove_columns=val_ds.column_names)

r = 32

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=r,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, config)
model.enable_input_require_grads()

args = TrainingArguments(
    output_dir=f"./output/Qwen3_lora_r{r}_homo",
    eval_strategy="steps",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()