from transformers import AutoTokenizer
import pandas as pd
from transformer import Transformer, Transformer_Trainer, Transformer_Dataset
import torch
from torch.utils.data import DataLoader



checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

batch_size = 4

df = pd.read_csv("./Data_Process/mbti_1.csv")

token_mbti = {"[PAD]" : 0, "[CLS]" : 1, "[SOS]" : 2, "I" : 3, "E" : 4, "N" : 5, "S" : 6, "F" : 7, "T" : 8, "J" : 9, "P" : 10}

for i in range(len(df["type"])):
    df["type"][i] = [1, token_mbti[df["type"][i][0]], token_mbti[df["type"][i][1]], token_mbti[df["type"][i][2]], token_mbti[df["type"][i][3]], 2]


data = [(df["posts"][i], df["type"][i]) for i in range(len(df["posts"]))]

print(df["type"][0])

dataset = Transformer_Dataset(data, tokenizer, seq_len=128)



train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, pin_memory=True)




src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = 11 # [PAD], [SOS], [CLS], I, E, N, S, F, T, J, P
d_model = 128
num_heads = 8
num_layers = 6
d_ff = 512
max_seq_length = 128
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device = "cuda")

transformer.cuda()

prev_epochs = 0
epochs = 3


# num_training_steps = epochs * (num_data_sample/batch_size)
trainer = Transformer_Trainer(transformer, train_loader, num_warmup_steps= 1000, num_training_steps = epochs * (len(dataset)/batch_size),
                              padding_token=0, device='cuda')



for epoch in range(prev_epochs, epochs):
    trainer.train(epoch)
    torch.save(transformer.state_dict(), "Trained_Model/transformer_1")
