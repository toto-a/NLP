import  os
import torch
import torch.nn as nn
from tqdm import tqdm
from string import punctuation
from sklearn.model_selection import train_test_split
from model import SimpleRNN

from utils import tensor_names, target_langs, lang2label, label2lang, name2tensor, num_letters




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


def train(model, criterion, optimizer, dataset, n_epochs=2, name2tensor=name2tensor):
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




if __name__ == "__main__":
    hidden_size = 256
    learning_rate = 0.001

    model= SimpleRNN(num_letters, hidden_size, len(lang2label))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, criterion, optimizer, train_dataset, n_epochs=2)
         
    print("Training finished")