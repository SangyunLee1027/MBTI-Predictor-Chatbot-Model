import torch
import torch.nn as nn
import torch.optim as optim 
import math
import re
from torch.utils.data import DataLoader, Dataset
import tqdm



class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim, dropout = 0):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.activation = torch.nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, device = "cuda"):
        super().__init__()
        pe = torch.zeros((max_seq_length, d_model))
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
    
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, middle_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first = True)
        self.feed_forward = FeedForward(d_model, middle_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, target_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, target_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device = "cuda:0"):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model).to(device)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        self.device = device

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask


    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        temp = self.encoder_embedding(src)
        src_embedded = self.dropout(self.positional_encoding(temp))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


class Transformer_Trainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr= 1e-4,
        device='cuda'
        ):

        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr = lr)

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)


    def iteration(self, epoch, data_loader, train = True):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            
            # 1. forward the input data to get output
            output = self.model.forward(data["Encoder_input"], data["Decoder_input"])
            
            # print(winner_output.shape)
            # 2-1. Crossentroyp loss of winner classification result
            # loss = self.criterion(winner_output, (data["winner_label"]))
            loss = self.criterion(output, data["Decoder_output"])

            # 3. backward and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # next sentence prediction accuracy
            # correct = winner_output.argmax(dim=-1).eq(data["winner_label"]).sum().item()
            correct = torch.round(output).eq(data["Decoder_output"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["Decoder_output"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % 10 == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}, \
            total_acc={total_correct * 100.0 / total_element}"
        ) 

class Transformer_Dataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=512):
        
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.corpus_lines = len(data_pair)
        self.data = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        encoder_input, decoder_inoutput = self.data[item]
        
        encoder_input = self.tokenizer(self.remove_urls(encoder_input), padding='max_length')['input_ids'][:512]
        
        tokened_res = self.tokenizer(" ".join(decoder_inoutput))['input_ids']
        decoder_input = tokened_res[:-1] + [0] * 507
        decoder_output = tokened_res[1:] + [0] * 507

        output = {"Encoder_input" : encoder_input, "Decoder_input" : decoder_input, "Decoder_output" : decoder_output}


        return {key: torch.tensor(value) for key, value in output.items()}
    

    def remove_urls(self, text):
        # Regex pattern to match URLs
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
