import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TLDRDataset(torch.utils.data.Dataset):
    def __init__(self, split, tok_model, batch_size = 1, max_length_post=500, max_length_sum=50):
        dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split=split)
        self.post = [t.get('post') for t in dataset['info']]
        self.pos = [t[ch].get('text') for t, ch in zip(dataset['summaries'], dataset['choice'])]
        self.neg = [t[1-ch].get('text') for t, ch in zip(dataset['summaries'], dataset['choice'])]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model, trust_remote_code=True)
        self.max_length_post = max_length_post
        self.max_length_sum = max_length_sum
        self.len = len(self.post)
        self.idx = np.arange(self.len)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        enc_post = self.tokenizer(
            [self.post[i] for i in idx], 
            truncation=True, max_length=self.max_length_post, padding="max_length"
        )
        enc_pos = self.tokenizer(
            [self.pos[i] for i in idx], 
            truncation=True, max_length=self.max_length_sum, padding="max_length"
        )
        enc_neg = self.tokenizer(
            [self.neg[i] for i in idx], 
            truncation=True, max_length=self.max_length_sum, padding="max_length"
        )
        ones = torch.ones(len(idx),1)
        return torch.tensor(enc_post['input_ids']), torch.tensor(enc_pos['input_ids']), torch.tensor(enc_neg['input_ids']), \
    torch.cat([torch.tensor(enc_post['attention_mask']), ones, torch.tensor(enc_pos['attention_mask'])], dim=-1), \
    torch.cat([torch.tensor(enc_post['attention_mask']), ones, torch.tensor(enc_neg['attention_mask'])], dim=-1)
    
    def shuffle(self):
        self.idx = np.random.permutation(self.idx)
        
    def get_batch(self):
        for i in range(0, self.len, self.batch_size):
            these = self.idx[i:i + self.batch_size]
            yield self.__getitem__(these)

class BindEmbeddings(nn.Module):
    def __init__(self, model, max_length_sum = 50):
        super().__init__()
        self.type_embedding = nn.Parameter(torch.rand(1, 1, model.embed_tokens.embedding_dim)) 
        self.model = model.embed_tokens
        self.summary_length = max_length_sum

    def forward(self, x):
        b = x.shape[0]
        return torch.cat([self.model(x[:,:-self.summary_length]), self.type_embedding.repeat(b,1,1), \
                          self.model(x[:,-self.summary_length:])], dim=1)

    
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model)
        self.config = self.model.config
        self.model.set_input_embeddings(BindEmbeddings(self.model))
        self.ff = nn.Sequential(
            nn.Linear(self.config.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
            )
    
    def tower(self, post, summary, mask):
        x = self.model(torch.cat([post, summary], dim=1), mask)
        x = self.ff(x.last_hidden_state[:,-1,:])
        return x.squeeze(-1)
    
    def forward(self, post, pos, neg, pos_mask, neg_mask):
        pos_score = self.tower(post, pos, pos_mask)
        neg_score = self.tower(post, neg, neg_mask)
        return pos_score - neg_score
        
class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, x):
        return torch.mean(torch.log(1+torch.exp(-x)))
    
### how can I squeeze out all singggleton dims    
    
#         # post_embedding = self.model.embed_tokens(post)
#         # summary_embedding = self.model.embed_tokens(summary)
#         # x = torch.cat([post_embedding, self.type_embedding.repeat(b,1,1), summary_embedding], dim=1)  
#         # self.model.set_input_embeddings(x)
#         # return self.model.get_input_embeddings()
#         # # position_embeddings = self.model.rotary_emb(x, torch.arange(seqlen).unsqueeze(0)) 
#         # # print(mask.shape)
#         # # print(x.shape)
#         # # x = self.model.layers[0](hidden_states = x, attention_mask = mask, position_embeddings = position_embeddings) 
#         # # return x 
         

# ds = TLDRDataset(split = "train", tok_model = "Qwen/Qwen3-0.6B-Base", batch_size=3 )
# for post, pos, neg, mask_pos, mask_neg in ds.get_batch():
#     break
# # mod =  AutoModel.from_pretrained(base_model)

# model = RewardModel("Qwen/Qwen3-0.6B-Base")
# xx = model(post, pos, neg, mask_pos, mask_neg)








