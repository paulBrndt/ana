import torch
from model import BigramLanguageModel

import time
import os

#hyperdata
batch_size = 64
block_size = 256
max_iters = 5000
training_percentage = 0.9
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


#load Training data
with open("train.txt") as file:
    data = file.read()

#defining all chars that are used in text
chars = sorted(list(set(data)))
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}")

#assigning each char an index
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

#and define functions to encode and decode string-index
def encode_string(string: str):
    encoded_list = [char_to_index[char] for char in string]
    return encoded_list


def decode_list(index_list: list[int]) -> str:
    decoded_string = "".join([index_to_char[index] for index in index_list])
    return decoded_string


#convert training data to tensor
data = torch.tensor(encode_string(data), dtype=torch.long)
n = int(training_percentage*len(data))
train = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get a random value
    x = torch.stack([data[i:i+block_size] for i in ix]) # the first block size (context)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # the target
    return x.to(device), y.to(device)

def generate(model, context="", max_new_tokens=300) -> str:
    context = torch.tensor(encode_string(context), dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode_list(generated)




model = BigramLanguageModel(vocab_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model_path = "main.pt"
model.load_state_dict(torch.load(model_path, map_location=device))

start_time = time.time()
version = str(start_time)

loss_data: list[tuple] = []
min_loss: float = 10



for iter in range(max_iters):
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
    loss_data.append((iter, time_diff, loss))
    
    #save stuff
    if (iter % 100) == 0:
        os.makedirs(f"./models/{version}", exist_ok=True)
        torch.save(model.state_dict(), f"./models/{version}/model-{str(iter)}-{str(loss)}.pt")
        print(f"Time since start: {time.time()-start_time}")
    if (iter % 50) == 0:
        with open("input.txt") as file:
            input = file.read()
        print(generate(model, "Hi there!" if input == "" else input, 50))
    

torch.save(model.state_dict(), f"./models/{version}/model-{str(iter)}-{str(loss)}")
torch.save(model.state_dict(), f"main.pt")
    
    



