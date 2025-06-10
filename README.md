1- finetuning_unsloth.py
Burada unslot kullanılarak model eğitimi yapılıyor. Unsloth colabdan alınıp yerelde çalışacak şekilde dizayn edildi. Ayrıca burada vast.ai da bir kez çalışması test edilmiş ve başarılı bir şekilde çalışması sonuçlanmıştır. Model hugginface e kaydedilip ollamaya kurulmuştur.

2-cpuFineTune_lora_gguf.py
Bu da 1. nin farklı bir hali aslında. Bunu hiç kullanmadık. Bunu yapmamızın nedeni sadece cpu da çalıştırmamız gerektiği için. Gpu şimdilik olmadığı için sunucuda çalıştıracağız. Burada yine unsloth kullanıldığı için ve unsloth kullanımında gpu zorunlu olduğu için bunu kullanamdık.

3-cpuFineTune_lora.py
Burada sadece cpu ile modeli hugginfacee entegre bir şekilde eğitebiliyoruz. Dataset ve oluşan model hugginface e gidiyoır ve kullanıabilir oluyor. Yanlız lora ile eğitildiği için dışarıya düşük boyutlu bir model çıkıyor ve bu modeli ollama ile entegre kullanamıoyurz. Belgede detaylara yer verildi. Bu modeli kullanamk için base model ile entegre kullanmak gerekiyor.

4-merge-models-base-lora.py
Burada modeli hugginface gguf formatında yükleyerek ve tek bir model olarak yükleyerek olamada kullanabilir olmayı hedefliyoruz. Onun için 3-cp... da oluşturduğumuz lora modeli ile base modeli merge ediyoruz ve ortaya tek bir model çıkıyor tabii yine gguf formatında değil henüz.

-------------
GGUF formatına dönüştürme
gereksinimler:
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt





NOT: 
Bunlarla alakalı detaylı bilgiler repo içerisindeki word dosyasında mevcut olacaktır.
