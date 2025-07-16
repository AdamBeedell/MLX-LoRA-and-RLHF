import torch
from reward_model import TLDRDataset, RewardModel, BradleyTerryLoss

# set random seed
torch.backends.cudnn.deterministic = True 
torch.manual_seed(1234)
torch.cuda.manual_seed_all(5678) 

config = {
    "base_model": "Qwen/Qwen3-0.6B-Base",
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "batch_size": 2,
    "learning_rate": 0.001,
    "num_epochs": 1
    }


def make(conf):
    train_ds = TLDRDataset(split = "train", tok_model = conf["base_model"], batch_size = conf["batch_size"])
    test_ds = TLDRDataset(split = "validation", tok_model = conf["base_model"], batch_size = conf["batch_size"])
    model = RewardModel(conf["base_model"])
    loss_fn = BradleyTerryLoss()
    return train_ds, test_ds, model, loss_fn  

def train(data, model, loss_fn, conf): 
    model.train() 
# do not update the encoder parameters
    for nam, par in model.named_parameters():
        if nam.startswith('model.embed_tokens.type_embedding'):
            par.requires_grad = True
        elif nam.startswith('model.layers.27.') or nam.startswith('model.norm.weight'):
            par.requires_grad = True
        elif nam.startswith('ff'):
            par.requires_grad = True    
        else:
            par.requires_grad = False
    model.to(conf['device'], non_blocking=True) 
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'])
    tr_losses = []
    for epoch in range(conf['num_epochs']):
        data.shuffle()
        losses = []
        for post, pos, neg, pos_mask, neg_mask in data.get_batch():
            optimizer.zero_grad()
            post = post.to(conf['device'], non_blocking=True)
            pos = pos.to(conf['device'], non_blocking=True)
            neg = neg.to(conf['device'], non_blocking=True)
            pos_mask = pos_mask.to(conf['device'], non_blocking=True)
            neg_mask = neg_mask.to(conf['device'], non_blocking=True)        
            diff = model(post, pos, neg, pos_mask, neg_mask)
            loss = loss_fn(diff)
            loss.backward()
            optimizer.step()
            losses.append(loss) 
            print(f'Train Loss: {loss:.4f}')
        mean_loss = sum(losses)/len(losses)
        print(f'Average train loss in epoch {epoch}: {mean_loss:.4f}')
    torch.save(model.state_dict(), "temp/current_model.pth")
    torch.save(torch.tensor(tr_losses), "temp/train_losses.pt")
    model.push_to_hub("ayzor/reward_model",
                      private=True, 
                      )

def test(data, model, loss_fn, conf): 
    model.eval()  
    model.to(conf['device'], non_blocking=True)
    losses = []
    for post, pos, neg, pos_mask, neg_mask in data.get_batch():
        with torch.no_grad():
            post = post.to(conf['device'], non_blocking=True)
            pos = pos.to(conf['device'], non_blocking=True)
            neg = neg.to(conf['device'], non_blocking=True)
            pos_mask = pos_mask.to(conf['device'], non_blocking=True)
            neg_mask = neg_mask.to(conf['device'], non_blocking=True)        
            diff = model(post, pos, neg, pos_mask, neg_mask)
            loss = loss_fn(diff) 
            losses.append(loss)
    mean_loss = sum(losses)/len(losses)
    print(f'Average test loss: {mean_loss:.4f}')
    torch.save(torch.tensor(losses), "temp/test_losses.pt") 

train_ds, test_ds, model, loss_fn = make(config)

# for post, pos, neg, pos_mask, neg_mask in train_ds.get_batch():
#     break

# x = model(post, pos, neg, pos_mask, neg_mask)
# lo = loss_fn(x)

# params = [nam for nam, _ in model.named_parameters()]
train(train_ds, model, loss_fn, config)
test(test_ds, model, loss_fn, config)











