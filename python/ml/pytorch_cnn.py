import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(in_features=28 * 28,
                            out_features=10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.fc = nn.Linear(in_features=28 * 28 * 10 // (2 * 2),
                            out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def classify(model, x):
    '''
    :param model:    network model object
    :param x:        (batch_sz, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_sz, ) torch tensor with class labels
    '''
    return torch.argmax(model(x), dim=1)


def get_model_class(_):
    ''' Do not change, needed for AE '''
    return [MyNet]


def train():
    batch_sz = 64
    learning_rate = 0.1
    epochs = 5

    dataset = datasets.MNIST('data', train=True, download=True,
                             transform=transforms.ToTensor())
    # trainTransform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
    #                                        transforms.ToTensor()])
    # dataset = datasets.ImageFolder('data_noise\\training', transform=trainTransform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                      [train_size,
                                                                       val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_sz,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(validation_dataset,
                                             batch_size=batch_sz,
                                             shuffle=True)

    device = torch.device("cpu")
    model = MyNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        for i_batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            net_output = model(x)
            loss = F.nll_loss(net_output, y)
            loss.backward()
            optimizer.step()

            if i_batch % 100 == 0:
                print('Train epoch: {}, batch: {}\tLoss: {:.4f}'.format(
                    epoch, i_batch, loss.item()))

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                net_output = model(x)

                prediction = classify(model, x)
                correct += prediction.eq(y).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        print('Validation accuracy: {:.2f}%'.format(100 * val_accuracy))

        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    train()
