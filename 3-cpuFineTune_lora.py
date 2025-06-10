# %%capture
# Gerekirse aşağıdaki satırların başındaki yorumları kaldırarak pip install yapabilirsiniz:
#
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# pip3 install transformers datasets peft trl accelerate
# pip3 install huggingface_hub

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

# -----------------------------------------------------------
# 1. Genel Ayarlar (CPU-Only)
# -----------------------------------------------------------
device = torch.device("cpu")
print("[MODEL İNDİRİLİYOR]")
MODEL_NAME = "microsoft/phi-4"
MAX_SEQ_LENGTH = 2048

# -----------------------------------------------------------
# 2. Tokenizer ve Modeli İndirme (CPU-Only)
# -----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)

# -----------------------------------------------------------
# 3. LoRA (PEFT) Ayarları
# -----------------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                     # CPU’da 8 veya 4 de tercih edilebilir
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.to(device)

# -----------------------------------------------------------
# 4. Dataset’i Yükleme ve Tokenizasyon
# -----------------------------------------------------------
print("[VERİ SETİ İNECEK]")
raw_dataset = load_dataset("mrkswe/llmEndpointDatasetConversation_2", split="train")
print("[VERİ SETİ İNDİRİLDİ]")

import json

def parse_conversations(example):
    example["conversations"] = json.loads(example["conversations"])
    return example

raw_dataset = raw_dataset.map(parse_conversations, remove_columns=[])

def convert_to_text(examples):
    texts = []
    for conv in examples["conversations"]:
        single_text = ""
        for turn in conv:
            speaker, txt = turn
            if speaker.lower() in ["user", "human"]:
                single_text += f"<s>user: {txt}\n"
            else:
                single_text += f"assistant: {txt}</s>\n"
        texts.append(single_text)
    return {"text": texts}

dataset_with_text = raw_dataset.map(convert_to_text, batched=True, remove_columns=["conversations"])

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

tokenized_dataset = dataset_with_text.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# -----------------------------------------------------------
# 5. Data Collator (Causal LM için)
# -----------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # “mlm=False” deyince otoregresif (causal) moda geçer
)

# -----------------------------------------------------------
# 6. TrainingArguments (CPU-Only)
# -----------------------------------------------------------
print("[HİPERPARAMETRELER HAZIRLANIYOR]")
training_args = TrainingArguments(
    output_dir="./outputs_cpu",
    per_device_train_batch_size=1,      # CPU RAM’i kısıtlıysa küçük batch
    gradient_accumulation_steps=4,      # Efektif batch size = 1*4 = 4
    warmup_steps=5,
    max_steps=30,                       # Örnek amaçlı kısa eğitim
    learning_rate=2e-4,
    logging_steps=1,
    fp16=False,
    bf16=False,
    optim="adamw_torch",                # CPU uyumlu optimizer
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    no_cuda=True,                       # CPU-only modunu zorla
    report_to="none",
)

# -----------------------------------------------------------
# 7. Eğitim Başlatma
# -----------------------------------------------------------
print("[EĞİTİM BAŞLIYOR]")
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    args=training_args,
)

trainer_stats = trainer.train()

# -----------------------------------------------------------
# 8. Modeli Cihaza Kaydetme
# -----------------------------------------------------------
print("[MODELİ CİHAZA KAYDET]")
save_dir = "./fine_tuned_cpu_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# -----------------------------------------------------------
# 9. Hugging Face Hub’a Yükleme (Opsiyonel)
# -----------------------------------------------------------
print("[MODELİ HUGGINGFACE’E KAYDET]")
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=save_dir,
    path_in_repo=".",
    repo_id="mrkswe/llmPhi4Try_CPU",  # Kendi kullanıcıAdınız/modelAdınız
    repo_type="model",
    token="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"
)

print("Fine‑tuning tamamlandı. Model ve tokenizer klasöre kaydedildi:", save_dir)



# --------------------------------------
# AÇIKLAMALAR LORA VS.
# --------------------------------------

# 1. Neden sadece adapter_model.safetensors dosyası var ve neden bu kadar küçük?
# Sen sadece LoRA adaptörleri ile fine-tuning yapıyorsun, yani:
#
# Ana model (phi-4, ~1.8B parametre) değiştirilmiyor.
#
# Sadece birkaç milyon parametre içeren LoRA katmanları eğitiliyor ve kaydediliyor.
#
# Dolayısıyla:
#
# adapter_model.safetensors dosyası sadece bu küçük farkları içeriyor.
#
# Bu yüzden boyutu küçük: 26.2 MB (gayet normal).
#
# Not: Eğer ana modeli de değiştirmek isteseydin (full fine-tuning), 6–7 GB gibi büyük bir dosya oluşurdu.

