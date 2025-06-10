# %%capture
# Gerekirse aşağıdaki satırların başındaki yorumları kaldırarak pip install yapabilirsiniz:
#
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# pip3 install transformers datasets peft trl accelerate
# pip3 install huggingface_hub

#

import os
import json
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template



# ------------------------------------------
# 1. MODEL YÜKLEME (CPU + LoRA destekli)
# ------------------------------------------
print("1. MODEL YÜKLEME (CPU + LoRA destekli)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",  # GGUF destekli Unsloth versiyonu
    max_seq_length=2048,
    load_in_4bit=False,  # CPU için False önerilir
)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=False,
    use_rslora=False,
    loftq_config=None,
)

# ------------------------------------------
# 2. TOKENIZER CHAT TEMPLATE
# ------------------------------------------
print("2. TOKENIZER CHAT TEMPLATE")
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

# ------------------------------------------
# 3. VERİ SETİ YÜKLEME VE DÖNÜŞTÜRME
# ------------------------------------------
print("3. VERİ SETİ YÜKLEME VE DÖNÜŞTÜRME")
dataset = load_dataset("mrkswe/llmEndpointDatasetConversation_2", split="train")

def parse_conversations(example):
    example['conversations'] = json.loads(example['conversations'])
    return example

dataset = dataset.map(parse_conversations)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ------------------------------------------
# 4. EĞİTİM
# ------------------------------------------
print("4. EĞİTİM")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    packing=False,
    args=TrainingArguments(
        output_dir="./outputs_cpu",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        no_cuda=True,
        report_to="none",
    ),
)

trainer.train()

# ------------------------------------------
# 5. GGUF FORMATTA KAYDETME ve HUGGING FACE’E YÜKLEME
# ------------------------------------------
print("5. GGUF FORMATTA KAYDETME ve HUGGING FACE’E YÜKLEME")
from huggingface_hub import login
login(token="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")

# GGUF dosyasını kaydet ve HuggingFace’e gönder
model.push_to_hub_gguf(
    repo_id="mrkswe/llmPhi4Try_CPU",  # kendi repo adın
    tokenizer=tokenizer,
    quantization_method="q4_k_m",     # GGUF formatı
    token="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"
)



# --------------------------------------
# AÇIKLAMALAR LORA VS.
# --------------------------------------

# Burada gguf formatında işlemleri hf huba kaydedeceğiz ama ekran kartı olmadan dönüşümü yapamıyoruz.
# Eğer bu şekilde kaydedebilirsek lama ile indirip kurabilir hale gelebilirdik.
# EKRAN KARTI OLMADIĞI İÇİN FINETUNING_UNSLOTH.PY dosyasını çalıştıramadığım gibi bunu da çalıştıram ıyorum. Bir işe yaramıyor.
