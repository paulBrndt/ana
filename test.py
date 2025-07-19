import torch
from model import BigramLanguageModel

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

with open("train.txt") as file:
    data = file.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

model = BigramLanguageModel(vocab_size)
model.to(device)
print(f"device: {device}")
model_path = "models/1752776734.190778/model-4700-tensor(2.5562, device='mps:0', grad_fn=<NllLossBackward0>)"
model.load_state_dict(torch.load(model_path, map_location=device))
print("loaded")
torch.save(model.state_dict(), "models/main.pt")