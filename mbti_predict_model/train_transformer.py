from transformers import AutoTokenizer
import pandas as pd
from transformer import Transformer, Transformer_Trainer, Transformer_Dataset
import torch
from torch.utils.data import DataLoader



checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

df = pd.read_csv("./Data_Process/mbti_1.csv")
data = [(df["posts"][i], df["type"][i]) for i in range(len(df["posts"]))]
dataset = Transformer_Dataset(data, tokenizer)

train_loader = DataLoader(
    dataset, batch_size=1, shuffle=True, pin_memory=True)


src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = tokenizer.vocab_size
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 512
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device = "cpu")

trainer = Transformer_Trainer(transformer, train_loader, device='cpu')

prev_epochs = 0
epochs = 20


for epoch in range(prev_epochs, epochs):
    trainer.train(epoch)
    torch.save(transformer.state_dict(), "Trained_Model/transformer_1")
