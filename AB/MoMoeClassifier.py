## MoMoeClassifier.oy


import torch
print(torch.__version__)
print(torch.cuda.is_available())  ## looking for True


### Load dataset from classifierdata.csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer


import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim



# --- Load CSV ---
df = pd.read_csv("classifierdata.csv")  # columns: prompt, expert
prompts = df["prompt"].tolist()
labels = df["expert"].tolist()


# --- Vectorize Text (TF-IDF) ---
vectorizer = TfidfVectorizer(max_features=512)
X = vectorizer.fit_transform(prompts).toarray()
y = torch.tensor(labels, dtype=torch.long)

# --- Dataset and Loader ---
class PromptDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

dataset = PromptDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)



class MoMoEClassifier(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.fc1 = NN.Linear(512, 256)  ############################################################################ Figure out a good context window here
        self.fc2 = NN.Linear(256, 128)  # half as many nodes
        self.fc3 = NN.Linear(128, 64)   # half as many nodes
        self.fc4 = NN.Linear(64, 5) # Output layer (64 -> 5, one for each valid expert in the MoMoE model)

    def forward(self, x):  # feed forward
        x = F.relu(self.fc1(x))  # Normalization function (ReLU)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here, end of the road ("cross-entropy expects raw logits" - which are produced here, the logits will be converted to probabilities later by the cross-entropy function during training and softmax during training and inference)
        return x
    
loss_function = NN.CrossEntropyLoss()  # using built-in loss function


model = MoMoEClassifier() ##create the model as described abvoe

optimizer = optim.Adam(model.parameters(), lr=0.001) ### lr = learning rate, 0.001 is apparently a "normal" value. Adam is the optimizer chosen, also fairly default



##### do training

num_epochs = 15 ## passes through the dataset


for epoch in range(num_epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "MoMoEWeights.pth")
torch.save(model, "MoMoEClassifier.pth")

