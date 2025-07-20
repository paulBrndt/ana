# utils.py
with open("train.txt") as file:
    data = file.read()

chars = sorted(list(set(data)))

char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}
    
def get_training_data() -> str:
    return data

def get_vocab_size() -> int:
    return len(chars)

def encode_string(self, string: str) -> list[int]:
    encoded_list = [self.char_to_index[char] for char in string]
    return encoded_list

def decode_list(self, index_list: list[int]) -> str:
    decoded_string = [self.index_to_char[index] for index in index_list]
    return decoded_string