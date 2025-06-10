# CPU ile çıkardığımız parça lora modeli ile base modeli birleştirip tek bir model yapıyoruz

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1. Ana modeli yükle (Phi-4 tabanı)
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")

# 2. LoRA modelini yükle (senin fine-tune ettiğin klasör)
model = PeftModel.from_pretrained(base_model, "./fine_tuned_cpu_model")

# 3. LoRA ile birleşimi tamamla
merged_model = model.merge_and_unload()

# 4. Birleştirilmiş modeli kaydet
merged_model.save_pretrained("./merged_model")

# 5. Tokenizer'ı da aynı klasöre kaydet
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
tokenizer.save_pretrained("./merged_model")

