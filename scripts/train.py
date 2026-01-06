"""
LoRA を使用した Hugging Face モデルの微調整スクリプト
"""
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset


def load_jsonl(file_path: str) -> List[Dict]:
    """JSONL ファイルを読み込む"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_config(config_path: str) -> Dict:
    """YAML 設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(args):
    # 設定を読み込む
    config = load_config(args.config)
    
    print(f"設定: {config}")
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    
    # モデルとトークナイザーの読み込み
    model_name = config.get('model_name', 'rinna/japanese-gpt2-medium')
    print(f"モデル: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # pad_token が設定されていない場合は eos_token を使用
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None,
    )
    
    # LoRA の設定
    lora_config = LoraConfig(
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        target_modules=config.get('target_modules', ['c_attn']),
        lora_dropout=config.get('lora_dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # データの読み込み
    train_data = load_jsonl(args.train_data)
    valid_data = load_jsonl(args.valid_data)
    
    print(f"訓練データ: {len(train_data)} 件")
    print(f"検証データ: {len(valid_data)} 件")
    
    # チャット形式をテキストに変換する関数
    def format_chat_to_text(messages):
        """チャット形式のメッセージをトークン化可能な形式に変換"""
        formatted_text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                formatted_text += f"### User:\n{content}\n\n"
            elif role == 'assistant':
                formatted_text += f"### Assistant:\n{content}\n\n"
        return formatted_text.strip()
    
    # データを変換
    train_texts = [format_chat_to_text(d['messages']) for d in train_data]
    valid_texts = [format_chat_to_text(d['messages']) for d in valid_data]
    
    # トークン化処理
    def tokenize(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length',  # 最大長にパディング
        )
        # ラベルを設定（言語モデリングタスク）
        outputs['labels'] = outputs['input_ids'].copy()
        return outputs
    
    train_dataset = Dataset.from_dict({
        'text': train_texts
    }).map(tokenize, batched=True)
    
    valid_dataset = Dataset.from_dict({
        'text': valid_texts
    }).map(tokenize, batched=True)
    
    # 訓練設定
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
    )
    
    # データコレーターの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果的言語モデリング
    )
    
    # Trainer の初期化と訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    
    print("訓練を開始します...")
    trainer.train()
    
    # モデルの保存
    model.save_pretrained("./output/final_model")
    tokenizer.save_pretrained("./output/final_model")
    print("モデルを保存しました: ./output/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lora.yaml')
    parser.add_argument('--train_data', type=str, default='data/train.jsonl')
    parser.add_argument('--valid_data', type=str, default='data/valid.jsonl')
    
    args = parser.parse_args()
    main(args)
