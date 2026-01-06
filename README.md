# hf-finetune-hello

Hugging Face を使用したモデルの LoRA 微調整プロジェクトです。

## プロジェクト構成

```
hf-finetune-hello/
├─ .venv/                # 仮想環境（Git 管理しない）
├─ data/                 # トレーニングデータ
│  ├─ train.jsonl       # 訓練データ
│  └─ valid.jsonl       # 検証データ
├─ scripts/              # メインスクリプト
│  ├─ train.py          # 微調整用スクリプト
│  └─ infer.py          # 推論用スクリプト
├─ configs/              # 設定ファイル
│  └─ lora.yaml         # LoRA 設定
├─ requirements.txt      # Python 依存関係
├─ .gitignore           # Git 管理除外リスト
└─ README.md            # このファイル
```

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv .venv
```

### 2. 仮想環境の有効化

**Windows (PowerShell):**
```bash
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bash
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### モデルの微調整

```bash
python scripts/train.py --config configs/lora.yaml --train_data data/train.jsonl --valid_data data/valid.jsonl
```

### 推論（テキスト生成）

```bash
python scripts/infer.py --model_path ./output/final_model --prompt "あなたのプロンプト"
```

対話型モード（プロンプト入力待ち）:
```bash
python scripts/infer.py --model_path ./output/final_model
```

## パラメータ設定

### train.py のパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--config` | `configs/lora.yaml` | 設定ファイルのパス |
| `--train_data` | `data/train.jsonl` | 訓練データのパス |
| `--valid_data` | `data/valid.jsonl` | 検証データのパス |

### infer.py のパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--model_path` | `./output/final_model` | 微調整済みモデルのパス |
| `--prompt` | None | 生成用プロンプト（省略時は入力待ち） |
| `--max_length` | 200 | 最大生成トークン数 |
| `--temperature` | 0.7 | 生成のランダム性（0～1） |
| `--top_p` | 0.9 | Nucleus サンプリングの p 値 |

## データフォーマット

訓練・検証データは JSONL 形式です。各行が JSON オブジェクトである必要があります。

**例 (data/train.jsonl):**
```json
{"text": "これは訓練データの例です。", "label": "example1"}
{"text": "機械学習モデルを微調整しています。", "label": "example2"}
```

## LoRA 設定のカスタマイズ

[configs/lora.yaml](configs/lora.yaml) を編集して設定をカスタマイズできます：

```yaml
# モデル指定
model_name: "gpt2"  # または "meta-llama/Llama-2-7b", など

# LoRA パラメータ
lora_r: 8           # 秩（小さいほどパラメータ削減）
lora_alpha: 16      # スケーリング係数
lora_dropout: 0.05  # Dropout 率

# 訓練設定
num_epochs: 3       # エポック数
batch_size: 4       # バッチサイズ
```

## 必要な環境

- Python 3.8 以上
- CUDA 対応 GPU（推奨）
- メインメモリ 16GB 以上

### GPU メモリが限られている場合

`configs/lora.yaml` の `mixed_precision: true` を有効にしてください。

## トラブルシューティング

### CUDA メモリ不足エラー

- `batch_size` を削減
- `max_length` を削減
- `mixed_precision: true` を設定

### モデルのダウンロード失敗

Hugging Face Hub へのインターネット接続を確認してください。

```bash
huggingface-cli login  # 必要に応じて
```

## ライセンス

MIT License

## 参考資料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning) Documentation](https://huggingface.co/docs/peft/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
