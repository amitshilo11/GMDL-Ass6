import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._C import device

###
# plot a scatter plot of coordinates with labels "labels"
# the data contain k classes
###
def scatter_plot(coordinates,labels,k):
    fig, ax = plt.subplots()
    for i in range(k):
        idx = labels == i
        data = coordinates[:, idx]
        ax.scatter(data[0], data[1], label=str(i), alpha=0.3, s=10)
    ax.legend(markerscale=2)
    plt.show()


def load_data(path):
    train = pd.read_csv(path)
    Y =  train['label']
    X = train.drop(['label'], axis=1)
    return X,Y

### FOR THE AUTOENCODER PART

def train(num_epochs, dataloader, model, criterion, optimizer):
    for epoch in range(num_epochs):
        running_loss, running_count = 0.0, 0

        for data in dataloader:
            img, _ = data
            # ===================forward=====================
            output, _ = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_count += 1

        # ===================log========================
        print('epoch [ {} / {} ], loss: {:.4f}'
              .format(epoch + 1, num_epochs, running_loss / len(dataloader)))

def get_embedding(model,dataloader):
    model.eval()
    labels = np.zeros((0,))
    embeddings = np.zeros((2,0))
    for data in dataloader:
        X,Y = data
        with torch.no_grad():
            _, code = model(X)
        embeddings = np.hstack([embeddings,code.numpy().T])
        labels = np.hstack([labels,np.squeeze(Y.numpy())])
    return embeddings,labels