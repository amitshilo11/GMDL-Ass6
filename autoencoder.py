# Import Libraries
import numpy as np
import torch
from torch import nn
import utils
from sklearn.preprocessing import StandardScaler
import sys

class autoencoder(nn.Module):
    def __init__(self):
        img_size = 28*28
        CL = 2
        EL1, EL2, EL3 = 288, 96, 32
        DL1, DL2, DL3 = 32, 96, 288
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size, EL1),
            nn.ReLU(True),
            nn.Linear(EL1, EL2),
            nn.ReLU(True),
            nn.Linear(EL2, EL3),
            nn.ReLU(True),
            nn.Linear(EL3, CL)
        )
        self.decoder = nn.Sequential(
            nn.Linear(CL, DL1),
            nn.ReLU(True),
            nn.Linear(DL1, DL2),
            nn.ReLU(True),
            nn.Linear(DL2, DL3),
            nn.ReLU(True),
            nn.Linear(DL3, img_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y, x

class linear_autoencoder(nn.Module):
    def __init__(self):
        img_size = 28*28
        CL = 2
        EL1, EL2, EL3 = 288, 96, 32
        DL1, DL2, DL3 = 32, 96, 288
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size, EL1),
            nn.Linear(EL1, EL2),
            nn.Linear(EL2, EL3),
            nn.Linear(EL3, CL)
        )
        self.decoder = nn.Sequential(
            nn.Linear(CL, DL1),
            nn.Linear(DL1, DL2),
            nn.Linear(DL2, DL3),
            nn.Linear(DL3, img_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y, x


def main():
    X, Y = utils.load_data(sys.argv[1])
    standardized_scalar = StandardScaler(with_std=False)
    X = standardized_scalar.fit_transform(X)
    X = np.array(X)                     # X[Data_Size, Pixels]
    Y = np.array(Y).reshape((-1, 1))    # Y[Labels]
    X = torch.tensor(X,dtype=torch.float32)
    Y = torch.tensor(Y,dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    num_epochs = 20
    num_of_labels = 10
    lerning_rate = 0.001
    criterion = nn.MSELoss()

    ############   Autoencoder Net   ###############
    Net = autoencoder()
    optimizer = torch.optim.Adam(Net.parameters(), lr=lerning_rate)

    utils.train(num_epochs, train_loader, Net, criterion, optimizer)
    embeddings, labels = utils.get_embedding(Net, train_loader)
    utils.scatter_plot(embeddings, labels, num_of_labels)


    #########   Autoencoder Linear Net   ############
    linear_net = autoencoder()
    linear_optimizer = torch.optim.Adam(linear_net.parameters(), lr=0.001)

    utils.train(num_epochs, train_loader, linear_net, criterion, linear_optimizer)
    embeddings, labels = utils.get_embedding(linear_net, train_loader)
    utils.scatter_plot(embeddings, labels, num_of_labels)

if __name__ == "__main__":
    main()