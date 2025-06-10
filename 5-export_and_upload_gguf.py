# modeli hugginface yüklemek içni


from huggingface_hub import HfApi

api = HfApi()

repo_id = "mrkswe/llmPhi4Try_CPU_2"  # senin repo ID'n
hf_token = "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"  # token

# yeni bir model reposu oluşturmak için
# api.create_repo(repo_id, exist_ok=True, repo_type="model")

api.upload_file(
    path_or_fileobj="phi4-finetuned-cpu.gguf",
    path_in_repo="phi4-finetuned-cpu.gguf",
    repo_id=repo_id,
    token=hf_token
)

print(f"GGUF dosyası yüklendi: https://huggingface.co/{repo_id}")

