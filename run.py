# run.py
import torch
from model import BigramLanguageModel

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

with open("train.txt") as file:
    data = file.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}")

char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

def encode_string(string: str):
    encoded_list = [char_to_index[char] for char in string]
    return encoded_list

def decode_list(index_list: list[int]) -> str:
    decoded_string = "".join([index_to_char[index] for index in index_list])
    return decoded_string

def generate(model: BigramLanguageModel, context="", max_new_tokens=300) -> str:
    context = torch.tensor(encode_string(context), dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=max_new_tokens, verbose=True)[0].tolist()
    return decode_list(generated)



model = BigramLanguageModel(vocab_size)
model.to(device)
print(f"device: {device}")
model_path = "main.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
print("loaded")
print(generate(model, "miranda"))

