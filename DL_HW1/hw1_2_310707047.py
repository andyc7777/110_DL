# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:10:55 2022

@author: Andy
"""

# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt 

# This is for the progress bar.
from tqdm import tqdm

#%%
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((32, 32)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

#%%
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("stanford_dogs_dataset/train", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("stanford_dogs_dataset/test", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#%%
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 8 * 8, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

#%%
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 80

# Whether to do semi-supervised learning.
do_semi = False

train_loss_record = []
test_loss_record = []
train_acc_record = []
test_acc_record = []

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    # if do_semi:
    #     # Obtain pseudo-labels for unlabeled data using trained model.
    #     pseudo_set = get_pseudo_labels(unlabeled_set, model)

    #     # Construct a new dataset and a data loader for training.
    #     # This is used in semi-supervised learning only.
    #     concat_dataset = ConcatDataset([train_set, pseudo_set])
    #     train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for idx, batch in loop:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    train_loss_record.append(train_loss)
    train_acc_record.append(train_acc)
    
    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Testing ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # # These are used to record information in validation.
    test_loss = []
    test_accs = []

    loop = tqdm(enumerate(test_loader), total=len(test_loader))

    for idx, batch in loop:

    #     # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        test_loss.append(loss.item())
        test_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_accs) / len(test_accs)
    test_loss_record.append(test_loss)
    test_acc_record.append(test_acc)

    # Print the information.
    print(f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")
#%%

def plot_result(config, train_loss_record, test_loss_record, train_acc_record, test_acc_record):

    # if config:
    #     epochs = config['epoch']
    # else:
    #     epochs = args.epochs

    # epochs = [i for i in range(epochs)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8))
    axes = axes.flatten()

    axes[0].set_title('Average Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Cross-entropy Loss')
    axes[0].plot(train_loss_record, label='Training')
    axes[0].plot(test_loss_record, label='Testing')
    axes[0].legend()

    axes[1].set_title('Total Accuracy')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Accuracy')
    axes[1].plot(train_acc_record, label='Training')
    axes[1].plot(test_acc_record, label='Testing')
    axes[1].legend()

    plt.show()

    return 

config = None
for i in range(len(train_acc_record)):
    train_acc_record[i] = train_acc_record[i].item()
    test_acc_record[i] = test_acc_record[i].item()

plot_result(config, train_loss_record, test_loss_record, train_acc_record, test_acc_record)