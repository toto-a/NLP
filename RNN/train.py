import  os
import torch
import torch.nn as nn
from tqdm import tqdm
from string import punctuation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as Dataloader
from model import SimpleRNN

from utils import tensor_names, target_langs, lang2label, label2lang, name2tensor, num_letters
from dataset import MyDataset



#####--------Name prediction language using RNN--------#####

train_idx, test_idx = train_test_split(
    range(len(target_langs)), 
    test_size=0.1, 
    shuffle=True, 
    stratify=target_langs
)

train_dataset = [
    (tensor_names[i], target_langs[i])
    for i in train_idx
]

test_dataset = [
    (tensor_names[i], target_langs[i])
    for i in test_idx
]

####--------Name prediction language using RNN--------#####



###### Name generation using RNN ######
text_data = open(os.path.join(os.path.dirname(__file__), "data/names.txt")).read()
text_dataset=MyDataset(text_data)
text_loader=Dataloader(text_dataset,batch_size=16,shuffle=True)
###### Name generation using RNN ######


def train_predict(model, criterion, optimizer, dataset, n_epochs=2, name2tensor=name2tensor):
    for epoch in tqdm(range(n_epochs)):
        for i,(name,label) in enumerate(dataset):
            hidden_state= model.init_hidden()

            for char in name:
                output, hidden_state = model(char, hidden_state)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if i % 10000 == 0:
                predicted=model.predict(name,label2lang)
                print(f"Predicted: {predicted}, Actual: {label2lang[label.item()]}")
                print(
                f"Epoch [{epoch + 1}/{n_epochs}], "
                f"Step [{i + 1}/{len(train_dataset)}], "
                f"Loss: {loss:.4f}"
            )


def train(model, optimizer, dataset_loader,loss_fn, n_epochs=4):

    epoch_loss={}
    for epoch in tqdm(range(n_epochs)):
        train_loss=list()
        for X,Y in dataset_loader : 

            if X.size(0)!=16:
                continue

            hidden=model.init_hidden()
            X,Y,hidden=X.float(),Y.float(),hidden.float()

            model.zero_grad()
            loss=0
            for c in range(X.size(1)):
                output,hidden=model(X[:,c].view(X.size(0),1),hidden)
                l=loss_fn(output,Y[:,c].long())
                loss+=l
            

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_loss.append(loss.detach().item()/X.size(1))
    
        epoch_loss[epoch]=torch.tensor(train_loss).mean().item()
        if epoch % 2 == 0:
                print(

                f"Epoch [{epoch + 1}/{n_epochs}], "
                f"Loss: {epoch_loss[epoch]:.4f}"
            )
                if not os.path.exists("model"):
                    os.mkdir("model/")
                torch.save(model.state_dict(), f"model/model_{epoch}.pth")
                print(model.generate(text_dataset))



        
    



if __name__ == "__main__":
    hidden_size = 256
    learning_rate = 0.001

    model= SimpleRNN(input_size=1 ,hidden_size=hidden_size,output_size=text_dataset.vocab_size) ## 1 char a time
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn=nn.CrossEntropyLoss()
    train(model, optimizer, text_loader,loss_fn, n_epochs=4)
         
    print("Training finished")