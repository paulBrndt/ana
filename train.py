# train.py
import torch

import time
import os

from model import BigramLanguageModel
from utils import *

#hyperdata
batch_size = 64
block_size = 256
max_iters = 6000
training_percentage = 0.9
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def prepare_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    data = get_training_data()
    data = torch.tensor(encode_string(data), dtype=torch.long)
    n = int(training_percentage*len(data))
    train = data[:n]
    val_data = data[n:]
    return (train, val_data)

train, val_data = prepare_training_data()

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get a random value
    x = torch.stack([data[i:i+block_size] for i in ix]) # the first block size (context)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # the target
    return x.to(device), y.to(device)

def generate(model: BigramLanguageModel, context="", max_new_tokens=300) -> str:
    context = torch.tensor(model.encode_string(context), dtype=torch.long).unsqueeze(0).to(device)
    return model.generate(context, max_new_tokens=max_new_tokens)

def test_model(model: BigramLanguageModel) -> str:
    with open("input.txt") as file:
        input = file.read()
    return generate(model, "Hi there!" if input == "" else input, 50)

def save_model(model: BigramLanguageModel):
    torch.save(model.state_dict(), f"main.pt")
    
def load_model(path: str) -> BigramLanguageModel:
    return torch.load(path, map_location=device)


model = BigramLanguageModel(get_vocab_size())
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model_path = "main.pt"
model.load_state_dict(load_model(model_path))

start_time = time.time()
loss_data: list[tuple] = []
min_loss: float = 10

counter_since_saved = 0

for iter in range(max_iters):
    counter_since_saved += 1
    start = time.time()

    #train stuff
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    #show stuff
    time_diff = time.time() - start
    min_loss = min(min_loss, float(loss))
    print(f"Iteration {iter:>4}, time: {time_diff:.1f}, loss {loss:.5f} (min loss: {min_loss:.5f})")
    loss_data.append((iter, time_diff, loss, (iter % 100) == 0))
    
    
    if (iter % 50) == 0:
        test_model(model) #generate some text to see output
        print(f"Time since start: {time.time()-start_time}")

        
    if (iter % 100) == 0 and (min_loss + .1) > loss:
        counter_since_saved = 0
        print("model saved")
        save_model(model)

    

save_model()    
    



