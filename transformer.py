import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pdb


class CodeDataset(Dataset):
    def __init__(self, tokens, tags, masks, device):
        super(CodeDataset, self).__init__()
        self.X = tokens  # .to(device)
        self.y = tags  # .to(device)
        self.masks = masks  # .to(device)
        # self.device = device

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]

    def __len__(self):
        return self.X.shape[0]


# dataset
class DataManager:
    def __init__(self, path, length=-1, device='cpu'):
        self.length, self.device = length, device
        self.token2idx, self.idx2token, self.tag2idx, self.idx2tag = self.init_vocabulary(f"{path}/train.txt")
        self.dataset = {part: self.read_data(f"{path}/{part}.txt") for part in ['train', 'valid', 'test']}

    def init_vocabulary(self, filepath):
        print(f"data init from {filepath}")
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if self.length == -1:
            self.length = max([len(line.split('\t')[0].split()) for line in lines])
        else:
            lines = [line for line in lines if len(line.split('\t')[0].split()) <= self.length]
        tokens = [token for line in lines for token in line.strip().split('\t')[0].split()]
        tags = [tag for line in lines for tag in line.strip().split('\t')[1].split()]
        idx2token = ['<PADDING>'] + [token for token in dict(Counter(tokens))][:-100] + ['<UNK>']
        token2idx = {token: idx for idx, token in enumerate(idx2token)}
        # idx2tag = ['<PADDING>'] + [tag for tag in dict(Counter(tags))]
        idx2tag = [tag for tag in dict(Counter(tags))]
        tag2idx = {tag: idx for idx, tag in enumerate(idx2tag)}
        # pdb.set_trace()
        return token2idx, idx2token, tag2idx, idx2tag

    def read_data(self, datafile):
        print(f"Reading {datafile}")
        with open(datafile, 'r') as f:
            codes = []
            tags = []
            for rawline in tqdm(f):
                code, tag = rawline.strip().split('\t')
                code, tag = code.split(), tag.split()
                if len(code) <= self.length and len(code) == len(tag):
                    codes.append(code)
                    tags.append(tag)

        code_ids = [torch.tensor(
            [self.token2idx[token] if token in self.token2idx else self.token2idx['<UNK>'] for token in code])
                    for code in tqdm(codes)]
        tag_ids = [
            torch.tensor([self.tag2idx[tag] if tag in self.tag2idx else self.O_tag_idx for tag in tag_seq]) for
            tag_seq in tqdm(tags)]
        masks = [torch.zeros_like(code) for code in tqdm(code_ids)]
        code_ids = pad_sequence(code_ids, padding_value=0, batch_first=True)
        tag_ids = pad_sequence(tag_ids, padding_value=0, batch_first=True)
        masks = pad_sequence(masks, padding_value=0, batch_first=True)
        if code_ids.shape[-1] < self.length:
            code_ids = torch.cat(
                [code_ids, torch.zeros((code_ids.shape[0], self.length - code_ids.shape[1]), dtype=torch.long)], dim=-1)
            tag_ids = torch.cat(
                [tag_ids, torch.zeros((tag_ids.shape[0], self.length - tag_ids.shape[1]), dtype=torch.long)], dim=-1)
            masks = torch.cat(
                [masks, torch.zeros((masks.shape[0], self.length - masks.shape[1]), dtype=torch.long)], dim=-1)
        # if code_ids.shape != tag_ids.shape or tag_ids.shape != masks.shape or code_ids.shape!=masks.shape:
        #     pdb.set_trace()
        # code_ids = torch.zeros((len(lines), self.length), dtype=torch.long)
        # tag_ids = torch.zeros((len(lines), self.length), dtype=torch.long)
        # masks = torch.zeros((len(lines), self.length), dtype=torch.long)
        #
        # for i, line in tqdm(enumerate(lines)):
        #     code, tags = line.split('\t')
        #     code = code.split()
        #     tags = tags.split()
        #     for j, (token, tag) in enumerate(zip(code, tags)):
        #         code_ids[i, j] = self.token2idx[token] if token in self.token2idx else self.token2idx['<UNK>']
        #         tag_ids[i, j] = self.tag2idx[tag] if tag in self.tag2idx else self.O_tag_idx
        #         # pdb.set_trace()
        #         masks[i, j] = 1
        return CodeDataset(code_ids, tag_ids, masks, self.device)

    def load(self, part, batch_size=128, shuffle=True):
        return DataLoader(dataset=self.dataset[part], batch_size=batch_size, shuffle=shuffle)

    @property
    def token_vocab_size(self):
        return len(self.idx2token)

    @property
    def tag_vocab_size(self):
        return len(self.idx2tag)

    @property
    def pad_tag_idx(self):
        return self.O_tag_idx  # self.tag2idx['<PADDING>']

    @property
    def O_tag_idx(self):
        if 'O' in self.tag2idx: return self.tag2idx['O']
        return self.tag2idx['0']


class TransformerConfig:
    dropout = 0

    def __init__(self,
                 token_vocab_size,
                 tag_vocab_size,
                 hidden_dim,
                 max_seq_len,
                 depth,
                 num_attention_heads
                 ):
        self.token_vocab_size = token_vocab_size
        self.tag_vocab_size = tag_vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        assert (hidden_dim % num_attention_heads == 0), "hidden_dim should be multiple of num_attention_heads"


class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(config.token_vocab_size, config.hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim))
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.register_parameter("position_embedding", self.position_embedding)

    def forward(self, inputs):
        """
        :param inputs: batch_size, max_seq_len
        :return: batch_size, max_seq_len, hidden_dim
        """
        return self.layer_norm(self.dropout(self.token_embedding(inputs) + self.position_embedding))


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_attention_heads
        self.scale = math.sqrt(config.hidden_dim)
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.attention_dropout = nn.Dropout(p=config.dropout)
        self.out_linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.resid_dropout = nn.Dropout(p=config.dropout)

    def forward(self, hidden_states, masks):
        """
        :param hidden_states: batch_size, max_seq_len, hidden_dim
        :param masks: batch_size, max_seq_len
        :return next_hidden_states: hidden_states: batch_size, max_seq_len, hidden_dim
        """
        bsz, msl, hdim = hidden_states.shape
        queries = self.query_proj(hidden_states).view(bsz, msl, self.num_heads, -1).transpose(1, 2)
        keys = self.key_proj(hidden_states).view(bsz, msl, self.num_heads, -1).transpose(1, 2)
        values = self.value_proj(hidden_states).view(bsz, msl, self.num_heads, -1).transpose(1, 2)

        attention_score = (queries @ keys.transpose(-1, -2)) / self.scale
        masks = masks.float()
        masks = (masks.unsqueeze(-1) @ masks.unsqueeze(1)).unsqueeze(1)  # bsz, 1, msl, msl
        attention = F.softmax(attention_score.masked_fill(masks == 0, -1e12), dim=-1)
        # pdb.set_trace()
        attention = self.attention_dropout(attention)
        next_hidden_states = (attention @ values).transpose(1, 2).reshape(bsz, msl, hdim)  # bsz, msl, hdim
        return self.resid_dropout(self.out_linear(next_hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(config)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.mlp_layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, hidden_states_n_masks):
        """
        :param hidden_states: batch_size, max_seq_len, hidden_dim
        :param masks: batch_size, max_seq_len
        :return: batch_size, max_seq_len, hidden_dim
        """
        hidden_states, masks = hidden_states_n_masks
        next_hidden_states = self.attention_layer_norm(self.multihead_attention(hidden_states, masks) + hidden_states)
        next_hidden_states = self.mlp_layer_norm(self.mlp(next_hidden_states) + next_hidden_states)
        return next_hidden_states, masks


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = TransformerEmbedding(config)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.depth)])
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc2 = nn.Linear(config.hidden_dim, config.tag_vocab_size)

    def forward(self, inputs, masks):
        """
        :param inputs: batch_size, max_seq_len
        :param masks: batch_size, max_seq_len
        :return: batch_size, max_seq_len, tag_vocab_size
        """
        embeddings = self.embedding(inputs)
        hidden_states, _ = self.transformer_blocks((embeddings, masks))

        self.fc1.weight.data = F.normalize(self.fc1.weight.data, p=2, dim=-1, eps=1e-5)
        self.fc2.weight.data = F.normalize(self.fc2.weight.data, p=2, dim=-1, eps=1e-5)
        hidden_states = self.gelu(self.fc1(hidden_states))
        return self.fc2(self.dropout(hidden_states))


def evaluate(prediction, gold, ignore_idx, O_idx):
    num_correct = (((prediction == gold).long()).masked_fill(gold == ignore_idx, 0)).sum().item()
    num_tags = ((gold != ignore_idx).long()).sum().item()
    print(num_correct, "/", num_tags)
    accuracy = num_correct / num_tags

    # num_correct = ((prediction == gold) & (gold != O_idx) & (gold != ignore_idx)).long().sum()
    # num_predict_positive = ((prediction != O_idx) & (gold != ignore_idx)).long().sum()
    # num_gold_positive = ((gold != O_idx) & (gold != ignore_idx)).long().sum()
    # pdb.set_trace()
    # precision, recall = num_correct / num_predict_positive, num_correct / num_gold_positive
    # f1 = (2 * precision * recall) / (precision + recall)
    # return namedtuple("evaluation_result", ["accuracy", "precision", "recall", "f1"]) \
    return accuracy.item(), 0,0,0 # precision.item(), recall.item(), f1.item()


def mean(lst):
    return sum(lst) / len(lst)


def train(model: Transformer, data: DataManager, batch_size: int, accumulation_steps: int, num_epochs: int,
          savepath: str, devices: list):
    best_loss = 999999
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model = nn.DataParallel(model, device_ids=devices)
    main_device = f"cuda:{devices[0]}"
    global_step = 0
    for epoch in range(num_epochs):
        train_losses = []
        model.module.train()
        for codes, tags, masks in tqdm(data.load('train', batch_size=batch_size)):
            codes, tags, masks = codes.to(main_device), tags.to(main_device), masks.to(main_device)
            predictions = model(codes, masks)
            bsz, msl = tags.shape
            # pdb.set_trace()
            loss = F.cross_entropy(predictions.view(bsz * msl, -1), tags.flatten(), ignore_index=data.pad_tag_idx)
            loss.backward()
            if (global_step + 1) % accumulation_steps == 0:
                for p in model.module.parameters() if (hasattr(model, 'module')) else model.parameters():
                    p.grad /= accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
            codes, tags, masks = codes.to('cpu'), tags.to('cpu'), masks.to('cpu')
            train_losses.append(loss.item())
            global_step += 1
        train_loss = mean(train_losses)

        model.module.eval()
        eval_losses, eval_predictions, eval_tags = [], [], []
        with torch.no_grad():
            for codes, tags, masks in tqdm(data.load('valid', batch_size=batch_size)):
                codes, masks, tags = codes.to(main_device), masks.to(main_device), tags.to(main_device)
                predictions = model(codes, masks)
                bsz, msl = tags.shape
                loss = F.cross_entropy(predictions.view(bsz * msl, -1), tags.flatten(), ignore_index=data.pad_tag_idx)
                codes, masks, tags = codes.to('cpu'), masks.to('cpu'), tags.to('cpu')
                eval_losses.append(loss.item())
                if epoch % 10 == 0:
                    eval_predictions.append(predictions.max(dim=-1)[1].to('cpu'))
                    eval_tags.append(tags)

        print(f"Epoch {epoch}:")
        eval_loss = mean(eval_losses)
        if epoch % 10 == 0:
            eval_predictions = torch.cat(eval_predictions, dim=0)
            eval_tags = torch.cat(eval_tags, dim=0)
            eval_acc, eval_prec, eval_rec, eval_f1 = evaluate(eval_predictions, eval_tags, data.pad_tag_idx,
                                                              data.O_tag_idx)
            print(
                f"Acc： {round(eval_acc, 4)} Eval Prec: {round(eval_prec, 4)}, Rec:{round(eval_rec, 4)}, F1 {round(eval_f1, 4)}"
            )
        print(f"Train Losses: {train_loss}, Eval Loss: {eval_loss}")

        if eval_loss < best_loss:
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), savepath)
            best_loss = eval_loss
            early_stop_buf = 0
        else:
            early_stop_buf += 1
            if early_stop_buf >= 3:
                break
    return model.module


def test(model: Transformer, data: DataManager, batch_size: int, savepath: str, devices: list):
    main_device = 'cpu' if len(devices) == 0 else f"cuda:{devices[0]}"
    model.load_state_dict(torch.load(savepath, map_location=main_device))
    model = nn.DataParallel(model, device_ids=devices)
    model.module.eval()
    test_losses, test_predictions, test_tags = [], [], []
    with torch.no_grad():
        for codes, tags, masks in data.load('test', batch_size=batch_size):
            codes, masks, tags = codes.to(main_device), masks.to(main_device), tags.to(main_device)
            predictions = model(codes, masks)
            bsz, msl = tags.shape
            loss = F.cross_entropy(predictions.view(bsz * msl, -1), tags.flatten(), ignore_index=data.pad_tag_idx)
            codes, masks, tags = codes.to('cpu'), masks.to('cpu'), tags.to('cpu')
            test_losses.append(loss.item())
            test_predictions.append(predictions.max(dim=-1)[1].to('cpu'))
            test_tags.append(tags)
    test_loss = mean(test_losses)
    test_predictions = torch.cat(test_predictions, dim=0)
    test_tags = torch.cat(test_tags, dim=0)
    test_acc, test_prec, test_rec, test_f1 = evaluate(test_predictions, test_tags, data.pad_tag_idx, data.O_tag_idx)
    print(
        f"Test Loss: {test_loss}, Test Acc {round(test_acc, 4)} Prec {round(test_prec, 4)} Rec {round(test_rec, 4)} F1 {round(test_f1, 4)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices")
    parser.add_argument("--dataset")
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--num_attention_heads", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--loss_accumulation_steps", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--savepath")
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    devices = list(map(int, args.devices.split(',')))
    main_device = f"cuda:{devices[0]}"
    data = DataManager(
        path=f"data/{args.dataset}",
        length=args.max_seq_len,
        device=main_device
    )

    model_config = TransformerConfig(
        token_vocab_size=data.token_vocab_size,
        tag_vocab_size=data.tag_vocab_size,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        depth=args.depth,
        num_attention_heads=args.num_attention_heads
    )
    model = Transformer(config=model_config).to(main_device)
    if args.train:
        model = train(model=model, data=data, batch_size=args.batch_size,
                      accumulation_steps=args.loss_accumulation_steps,
                      num_epochs=args.num_epochs, savepath=args.savepath, devices=devices)
    test(model, data, args.batch_size, args.savepath, devices)


if __name__ == '__main__':
    main()
