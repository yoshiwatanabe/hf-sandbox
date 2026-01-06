"""
微調整済みモデルを使用した推論スクリプト
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    
    # モデルとトークナイザーの読み込み
    print(f"モデルを読み込んでいます: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    model.eval()
    
    # プロンプトの処理
    prompt = args.prompt or input("プロンプトを入力してください: ")
    
    print(f"\nプロンプト: {prompt}\n")
    
    # トークン化と生成
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成結果:\n{result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./output/final_model')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    
    args = parser.parse_args()
    main(args)
