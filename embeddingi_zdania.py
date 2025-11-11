from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 1️⃣ Wybieramy model językowy (BERT – bardzo dobry do porównań semantycznych, wielojęzyczny - rozumie polski)
model_name = "bert-base-multilingual-cased"

# 2️⃣ Ładowanie tokenizera i modelu
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 3️⃣ Cztery zdania – te same słowa, różna kolejność
sentence1 = "Pies goni kota"
sentence2 = "Kota goni pies"
sentence3 = "Pies ściga kota"
sentence4 = "Kota ściga pies"

# 4️⃣ Tokenizacja i wektoryzacja
inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")
inputs3 = tokenizer(sentence3, return_tensors="pt")
inputs4 = tokenizer(sentence4, return_tensors="pt")

# 5️⃣ Obliczamy embeddingi (reprezentacje) zdań
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)
    outputs3 = model(**inputs3)
    outputs4 = model(**inputs4)

# 6️⃣ Bierzemy uśrednione wektory całych zdań
embedding1 = outputs1.last_hidden_state.mean(dim=1)
embedding2 = outputs2.last_hidden_state.mean(dim=1)
embedding3 = outputs3.last_hidden_state.mean(dim=1)
embedding4 = outputs4.last_hidden_state.mean(dim=1)

# 7️⃣ Obliczamy kosinusową podobieństwo (1.0 = identyczne, 0 = brak podobieństwa)
similarity1 = F.cosine_similarity(embedding1, embedding1)
similarity2 = F.cosine_similarity(embedding1, embedding2)
similarity3 = F.cosine_similarity(embedding1, embedding3)
similarity4 = F.cosine_similarity(embedding1, embedding4)

print(f"Zdanie 1: {sentence1}")
print(f"Zdanie 2: {sentence2}")
print(f"Zdanie 3: {sentence3}")
print(f"Zdanie 4: {sentence4}")
print(f"Podobieństwo kosinusowe 1-1: {similarity1.item():.4f}")
print(f"Podobieństwo kosinusowe 1-2: {similarity2.item():.4f}")
print(f"Podobieństwo kosinusowe 1-3: {similarity3.item():.4f}")
print(f"Podobieństwo kosinusowe 1-4: {similarity4.item():.4f}")
