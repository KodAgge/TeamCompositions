import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam, SGD
from utils.import_data import loadData

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(138, 64, kernel_size = (2, 1), padding = (1, 1))  # single matchup
        self.conv1_2 = nn.Conv2d(138, 64, kernel_size = (2, 3), padding = (1, 2))  # trio matchup
        self.conv1_3 = nn.Conv2d(138, 64, kernel_size = (2, 5), padding = (1, 3))  # full composition

        self.conv2 = nn.Conv2d(64 * 3, 256, kernel_size = 2, padding = (1, 1))

        self.conv3 = nn.Conv2d(256, 512, kernel_size = 2, padding = (1, 1))

        self.fc = nn.Linear(24576, 2) #  12288

    def forward(self, x):
        x = F.relu(torch.cat((self.conv1_1(x), self.conv1_2(x), self.conv1_3(x)), axis = 1))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))

        return x


def MGD(model, train_dataloader, optimizer, criterion, n_epochs):

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (X_train_batch, Y_train_batch) in enumerate(train_dataloader, 0):

            outputs = None
            
            # forward-pass
            outputs = model(X_train_batch)

            # backwards pass + gradient step
            loss = criterion(outputs, Y_train_batch.view(-1))
            
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward() #retain_graph=True)
            
            optimizer.step()

            if i == 0 and epoch == 0:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.99 * smooth_loss + 0.01 * loss.item()

            if i == 0 or (i+1) % 100 == 0:
                print("\tBatch", i+1, "of", len(train_dataloader), "complete")
                print("\t\tLoss =", loss.item())
                print("\t\tSmooth loss =", smooth_loss)

            running_loss += loss.item()


        print("\nEpoch", epoch + 1, "of", n_epochs, "complete")
        print("Average loss", running_loss / len(train_dataloader))
    
    print("=" * 50 + "\n\nTraining complete!")

    return model


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x
            y = y.view(-1)
            
            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()


model = Net()

optimizer = Adam(model.parameters(), lr=0.07)

criterion = CrossEntropyLoss()


trainLoader, testLoader, valLoader = loadData()

trained_model = MGD(model, trainLoader, optimizer, criterion, 1)

check_accuracy(trainLoader, trained_model)
check_accuracy(testLoader, trained_model)
check_accuracy(valLoader, trained_model)