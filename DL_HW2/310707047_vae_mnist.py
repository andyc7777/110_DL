#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:45:44 2022

@author: andychan
"""

import os
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


# Set random seed
seed = 999
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def grayscale_to_binary(data):
    threshold = 127
    binary_image = np.where(data >= threshold, 1.0, 0.0)
    return binary_image


class MnistDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data, self.target = self.__build_dataset__()

    def __build_dataset__(self):
        MNIST = np.load(self.root)
        data = MNIST['image'].astype(np.float32)
        target = MNIST['label'].astype(np.int32)

        # Grayscale to binary
        data[data > 0] = 1

        # Convert image from numpy to tensor
        data = torch.from_numpy(data).view(-1, 1, data.shape[1], data.shape[2])
        target = torch.from_numpy(target)

        return data, target

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return len(self.data)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_dim = 1 * 28 * 28
        self.layer_1 = 1024
        self.layer_2 = 512
        self.layer_3 = 256
        self.z_latent = 10

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.layer_1),
            nn.ReLU(),
            nn.Linear(self.layer_1, self.layer_2),
            nn.ReLU(),
            nn.Linear(self.layer_2, self.layer_3),
            nn.ReLU()
        )

        # Mean and variance
        self.mu = nn.Linear(self.layer_3, self.z_latent)
        self.var = nn.Linear(self.layer_3, self.z_latent)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.z_latent, self.layer_3),
            nn.ReLU(),
            nn.Linear(self.layer_3, self.layer_2),
            nn.ReLU(),
            nn.Linear(self.layer_2, self.layer_1),
            nn.ReLU(),
            nn.Linear(self.layer_1, self.input_dim),
            nn.Sigmoid()
            )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.var(h)

    def sampling(self, mean, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, var = self.encode(x.view(-1, self.input_dim))
        z = self.sampling(mean, var)
        return self.decode(z), mean, var


def get_dataloader(data_dir):

    # data preprocessing
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,),
                                                         std=(0.5,))])

    train_dataset = MnistDataset(root=data_dir, transform=transform)

    # Data Loader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=64,
                                  shuffle=False)

    return train_dataloader


def VAELoss(KL_lambda, reconstruct_x, x, mu, var):
    bce = F.binary_cross_entropy(reconstruct_x, x.view(-1, 784), reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    return bce + KL_lambda * KL_divergence


def draw_learning_curve(training_loss_record, KL_lambda):

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss_record, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./Results/mnist/mnist_learning_curve_KL_lambda_{}'.format(KL_lambda))
    plt.show()


def interpolate(KL_lambda, model, x_1, x_2, n=12):
    '''
    https://avandekleut.github.io/vae/
    '''
    mu_1, var_1 = model.encode(x_1)
    z_1 = model.sampling(mu_1, var_1)

    mu_2, var_2 = model.encode(x_2)
    z_2 = model.sampling(mu_2, var_2)

    z = torch.stack([t * z_1 + (1 - t) * z_2 for t in np.linspace(0, 1, n)])
    interpolate_list = model.decode(z)
    interpolate_list = interpolate_list.detach().cpu().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)

    plt.figure(figsize=(6*4, 4))
    plt.imshow(img)
    plt.title(r'Synthesized images of based on interpolation of two latent codes z ($\lambda$ = {})'.format(KL_lambda))
    plt.axis('off')
    plt.savefig('./Results/mnist/mnist_synthesised_img_lamda_{}.jpg'.format(str(KL_lambda)))

    return


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, '\n')

    train_dataloader = get_dataloader('./VAE_dataset/TibetanMNIST.npz')

    KL_lambda = 1

    model = VAE().to(device)

    # Load the model checkpoint if it exist
    model_PATH = './model_weight/mnist/model_weights_mnist_{}.pth'.format(KL_lambda)

    if os.path.exists(model_PATH):
        print('Model checkpoint exists.')
        model = torch.load(model_PATH)

    else:
        print('Model checkpoint does not exists, jump to training phase.')
        lr = 0.0001
        optimizer = optim.Adam(model.parameters(), lr)

        epochs = 500
        training_loss_record = []

        print('Training Phase...')  # Training
        for epoch in range(1, epochs+1):

            model.train()
            total_loss = 0

            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

            for i, data in loop:
                
                
                data = data[0].to(device)

                reconstruct_data, mu, var = model(data)
                loss = VAELoss(KL_lambda, reconstruct_data, data, mu, var)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 在progress bar內加入一些training stat的參數顯示以便觀察
                loop.set_description(f'Epoch [{epoch}/{500}]')
                loop.set_postfix({'Loss': '{:.6f}'.format(loss.item()/len(data))})

            # Averger loss of each epoch
            training_loss = total_loss / len(train_dataloader.dataset)
            training_loss_record.append(training_loss)

        draw_learning_curve(training_loss_record, KL_lambda)
        torch.save(model, './model_weight/mnist/model_weights_mnist_{}.pth'.format(KL_lambda))

    model.eval()

    with torch.no_grad():
        # Reconstrcuted sample
        data, _ = next(iter(train_dataloader))
        save_image(data.view(64, 1, 28, 28), './Results/mnist/mnist_real_data_KL_lambda_{}.jpg'.format(KL_lambda))

        reconstruct_data, _, _ = model(data.to(device))
        save_image(reconstruct_data.view(64, 1, 28, 28), './Results/mnist/mnist_reconstructed_data_KL_lambda_{}.jpg'.format(KL_lambda))

        # Sample z_latent and use it to synthesize examples
        z = torch.randn(64, 10).cuda()
        sample = model.decoder(z).cuda()
        save_image(sample.view(64, 1, 28, 28), './Results/mnist/mnist_sample_prior_data_KL_lambda_{}.jpg'.format(KL_lambda))

        # Interpolation
        data, target = train_dataloader.__iter__().next()
        x_1 = data[target == 9][1].view(-1, 784).to(device)
        x_2 = data[target == 4][1].view(-1, 784).to(device)
        interpolate(KL_lambda, model, x_1, x_2, n=20)

    # Visual results
    real_data = plt.imread('./Results/mnist/mnist_real_data_KL_lambda_{}.jpg'.format(KL_lambda))
    reconstructed_data = plt.imread('./Results/mnist/mnist_reconstructed_data_KL_lambda_{}.jpg'.format(KL_lambda))
    sample_data = plt.imread('./Results/mnist/mnist_sample_prior_data_KL_lambda_{}.jpg'.format(KL_lambda))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6*4, 4*4))  # (width, height)
    axes = axes.flatten()

    axes[0].set_title(r'Real samples in dataset ($\lambda$ = {})'.format(KL_lambda))
    axes[0].imshow(real_data)
    axes[0].axis('off')
    axes[1].set_title(r'Reconstructed sample using VAE ($\lambda$ = {})'.format(KL_lambda))
    axes[1].imshow(reconstructed_data)
    axes[1].axis('off')
    axes[2].set_title(r'Synthesized images of animation faces ($\lambda$ = {})'.format(KL_lambda))
    axes[2].imshow(sample_data)
    axes[2].axis('off')
    axes[3].axis('off')

    plt.savefig('./Results/mnist/mnist_KL_lambda_{}.jpg'.format(KL_lambda))
    plt.show()

if __name__ == '__main__':
    main()
