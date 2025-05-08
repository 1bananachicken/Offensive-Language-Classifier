import json
import torch
from tqdm import tqdm
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


r = 8
model_path = './weights/qwen/Qwen2.5-7B-Instruct/'
lora_path = f'./output/Qwen2.5_instruct_lora_r{r}_homo/checkpoint-600'

# model_path = './weights/Qwen/Qwen3-8B'
# lora_path = f'./output/Qwen3_lora_r{r}_homo/checkpoint-1200'

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=r,
    lora_alpha=32,
    lora_dropout=0.1
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
model.eval()


def predict(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个网络攻击性语言辨别程序，对于每一个输入，如果具有冒犯性，就输出1，否则输出0"},
        # {"role": "system", "content": "检测输入是否具有攻击性，1表示是，0表示否"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def load_data(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def evaluate(json_path: str):
    data = load_data(json_path)
    preds, labels = [], []

    for item in tqdm(data):
        prompt = item["instruction"]
        label = int(item["output"])
        pred = predict(prompt)

        preds.append(int(pred))
        labels.append(label)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary", pos_label=1)
    recall = recall_score(labels, preds, average="binary", pos_label=1)
    f1 = f1_score(labels, preds, average="binary", pos_label=1)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    evaluate("./proceed_dataset/homo/test.json")
    # evaluate("./proceed_dataset/emoji/test.json")
