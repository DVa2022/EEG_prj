import contextlib
import io
import torch
import os
import numpy as np
from aux_eegproj_funcs import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchviz
import mne





# DATASET
eeg_ds = EEGDataset(EEGTransform, exp_table=build_experiment_tbl('sleep_1_30sec.csv'))

# DATALOADERS
batch_sz = 8
# eeg_dl = DataLoader(eeg_ds, batch_size=batch_sz, shuffle=True, num_workers=0)
dl_train, dl_val, dl_test = train_val_test_split_by_pid(dataset=eeg_ds,
                                                        batch_size=batch_sz,
                                                        train_rt=.6,
                                                        val_rt=.2,  num_workers=0, verbose=1)
# Display Data
batch_to_show = 4
plt.figure(figsize=(18,6))
for i, (signals, labels) in enumerate(dl_train):  # signals.shape torch.Size([4, 2, 60000])
    labels = labels[0]
    for j in range(batch_sz):
        plt.subplot(batch_to_show, batch_sz, j+1+(i*batch_sz))
        plt.imshow(signals[j][:,:200])
        plt.title(labels[j].item())
        plt.axis('off')
    if i+1==batch_to_show:
        break
plt.show()


class GRUSex(nn.Module):
    def __init__(self, in_ax=1, aa_channels=128, out_ya=1, num_layers=2, batch_first=True, bidirectional=True):
        super().__init__()
        # ------Your code------#
        # Add GRU and linear layers.
        # Use the num_layer attribute to concatenate several GRU layers.
        # Use aa_channels to set the linear layer size according to the bidirectional input.

        self.GRU = nn.GRU(input_size=in_ax, hidden_size=aa_channels, num_layers=num_layers, batch_first=batch_first,
                          bidirectional=bidirectional)

        if bidirectional:
            in_features = aa_channels * 2
        else:
            in_features = aa_channels
        self.linear = nn.Linear(in_features=in_features, out_features=1)  # no activation - we perform regression.
        # ------^^^^^^^^^------#

    def forward(self, x, a_in=None):
        # ------Your code------#
        # Set a condition to use only x as input if a_in=None.
        if a_in is None:
            GRU_output, a_out = self.GRU(x)
        else:
            GRU_output, a_out = self.GRU(x, a_in)
        y_pred = self.linear(GRU_output)
        # ------^^^^^^^^^------#
        return y_pred, a_out


# Define a network to classify Sex:
class eegSexNet(nn.Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
        """
        super().__init__()
        # ------Your code------#
        # Define the CNN layers in a nn.Sequential.
        # Remember to use the number of input channels as the first layer input shape.
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=7, stride=1, padding=0, dilation=3),
            # TODO try changing the kernel sizes they were 3
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0, dilation=3),
            nn.ReLU(),
            Residual(in_channels=64)
        )

        # Compute the CNN output size here to use as the input size for the fully-connected part.
        CNN_forward = self.CNN(torch.zeros(input_shape))

        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation (why?).
        self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),
            # input shape is the flattened CNN output, output shape is 100
            nn.ReLU(),
            nn.Linear(100, 1),  # We need 1 neuron as an output. input shape is 100 output shape is 1
            nn.Sigmoid()
        )
        # ------^^^^^^^^^------#

    def forward(self, x):
        # ------Your code------#
        # Forward through the CNN by passing x, flatten and then forward through the linears.
        features = self.CNN(x)
        features = features.view(features.size(0), -1)  # reshape/flatten
        scores = self.FCs(features)
        # ------^^^^^^^^^------#
        return torch.squeeze(scores)

# Instantiate the network
eeg_sex_net = eegSexNet(signals.shape)
print('signal shape')
print(signals.shape)
print(eeg_sex_net)

# Loss Function for Sex eeg_sex_net
loss_function = nn.BCELoss()  # good for binary

# Optimizer for Sex eeg_sex_net
learning_rate = 0.0001
optimizer = torch.optim.Adam(params=eeg_sex_net.parameters(), lr=learning_rate)


# Training loop
def forward_epoch(model, dl, loss_function, optimizer, total_loss,
                  to_train=False, desc=None,device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        model = model.double().to(device)

        y_trues = torch.empty(0).type(torch.int)
        y_preds = torch.empty(0).type(torch.int)
        for i_batch, (X, y) in enumerate(dl):
            # print('sickyall',X.dtype)
            X = X.to(device)
            X = X.type(torch.double)
            # print('wackyall',X.dtype)
            y = y[0].to(device)  # added index because of get label returning sex, age

            # Forward:
            # print(X.shape)
            y_pred = model(X)

            # Loss:
            y_true = y.type(torch.double)
            loss = loss_function(y_pred, y_true)
            total_loss += loss.item()

            y_trues = torch.cat((y_trues, y_true))
            y_preds = torch.cat((y_preds, y_pred))
            if to_train:
                # Backward:
                optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
                loss.backward()  # get gradients

                # Optimization step:
                optimizer.step()  # use gradients

            # Progress bar:
            pbar.update(1)

    return total_loss, y_trues, y_preds

# Training SEXNET # TODO put into a function
epochs = 3

train_loss_vec = []
test_loss_vec = []
val_loss_vec = []
train_acc_vec = []
test_acc_vec = []
val_acc_vec = []
# GPU_0 = torch.device(0)
# print(GPU_0)
for i_epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    print(f'Epoch: {i_epoch + 1}/{epochs}')

    train_loss, y_true_train, y_pred_train = forward_epoch(eeg_sex_net, dl_train, loss_function, optimizer, train_loss,
                                                           to_train=True, desc='Train')  # , device=GPU_0)

    # Probably should be validation in real tasks...
    test_loss, y_true_test, y_pred_test = forward_epoch(eeg_sex_net, dl_test, loss_function, optimizer, test_loss,
                                                        to_train=False, desc='Test')  # , device=GPU_0)

    # Metrics:
    train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
    train_loss_vec.append(train_loss)
    train_accuracy = accuracy_score(y_true_train.cpu(), (
                y_pred_train.cpu().detach() > 0.5) * 1)  # scikit-learn computations are numpy based;thus should run on CPU and without grads.
    train_acc_vec.append(train_accuracy)

    test_loss = test_loss / len(dl_test)
    test_loss_vec.append(test_loss)
    test_accuracy = accuracy_score(y_true_test.cpu(), (y_pred_test.cpu().detach() > 0.5) * 1)
    test_acc_vec.append(test_accuracy)

    print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
          test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')

# PLOT SOME RESULTS FROM TRAINING SEXNET
plt.figure()
plt.plot(train_loss_vec, label='train')
plt.plot(test_loss_vec, label='test')
plt.ylabel('BCE loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_trues = torch.empty(0).type(torch.int)
y_preds = torch.empty(0).type(torch.int)
_, y_true_sex_val, y_pred_sex_val = forward_epoch(eeg_sex_net, dl_val, loss_function, optimizer, test_loss,
                                                  to_train=False,
                                                  desc='Test')  # , device=GPU_0) # run on the validation data

y_pred_sex_val = (y_pred_sex_val.cpu().detach()>0.5)*1
y_true_sex_val = y_true_sex_val.cpu()

ConfusionMatrixDisplay.from_predictions(y_true_sex_val, y_pred_sex_val, cmap='PuRd')
plt.show()

test_f1_score=f1_score(y_true_sex_val, (y_pred_sex_val))
print(f'test_f1-score={round(test_f1_score, 3)}')


# AGENET place in seperate file(?)

loss_function = nn.MSELoss()  # good for regression
learning_rate = 0.0001


# Define a network:
class eegAgeNet(nn.Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
        """
        super().__init__()
        # ------Your code------#
        # Define the CNN layers in a nn.Sequential.
        # Remember to use the number of input channels as the first layer input shape.
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=7, stride=1, padding=0, dilation=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0, dilation=3),
            nn.ReLU(),
            Residual(in_channels=64)
        )

        # Compute the CNN output size here to use as the input size for the fully-connected part.
        CNN_forward = self.CNN(torch.zeros(input_shape))

        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation (why?).
        self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 100),
            # input shape is the flattened CNN output, output shape is 100
            nn.ReLU(),
            nn.Linear(100, 1),  # We need 1 neuron as an output. input shape is 100 output shape is 1
        )
        # ------^^^^^^^^^------#

    def forward(self, x):
        # ------Your code------#
        # Forward through the CNN by passing x, flatten and then forward through the linears.
        features = self.CNN(x)
        features = features.view(features.size(0), -1)  # reshape/flatten
        scores = self.FCs(features)
        # ------^^^^^^^^^------#
        return torch.squeeze(scores)


agenet = eegAgeNet(signals.shape)
optimizer = torch.optim.Adam(params=agenet.parameters(), lr=learning_rate)


def forward_epoch(model, dl, loss_function, optimizer, total_loss, to_train=False, desc=None,
                  device=torch.device('cpu')):
    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        model = model.double().to(device)

        y_trues = torch.empty(0).type(torch.int)
        y_preds = torch.empty(0).type(torch.int)
        for i_batch, (X, y) in enumerate(dl):
            # print('sickyall',X.dtype)
            X = X.to(device)
            X = X.type(torch.double)
            # print('wackyall',X.dtype)
            y = y[1].to(device)  # TODO added because of get label returning sex, age

            # Forward:

            y_pred = model(X)
            # print('holler',X)
            # print('sfsd y_pred',y_pred) # all four predictions for batch of size four are the same
            # Loss:
            y_true = y.type(torch.double)
            loss = loss_function(y_pred, y_true)
            total_loss += loss.item()

            y_trues = torch.cat((y_trues, y_true))
            y_preds = torch.cat((y_preds, y_pred))
            if to_train:
                # Backward:
                optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
                loss.backward()  # get gradients

                # Optimization step:
                optimizer.step()  # use gradients

            # Progress bar:
            pbar.update(1)

    return total_loss, y_trues, y_preds  # get last batch with [-batch_size]


epochs = 3

train_loss_vec = []
test_loss_vec = []
train_acc_vec = []
test_acc_vec = []
# GPU_0 = torch.device(0)
# print(GPU_0)
for i_epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    print(f'Epoch: {i_epoch + 1}/{epochs}')

    train_loss, y_true_train, y_pred_train = forward_epoch(agenet, dl_train, loss_function, optimizer, train_loss,
                                                           to_train=True, desc='Train')  # , device=GPU_0)

    # Probably should be validation in real tasks...
    test_loss, y_true_test, y_pred_test = forward_epoch(agenet, dl_test, loss_function, optimizer, test_loss,
                                                        to_train=False, desc='Test')  # , device=GPU_0)

    # Metrics:
    train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
    train_loss_vec.append(train_loss)
    train_accuracy = accuracy_score(y_true_train.cpu(), (
                y_pred_train.cpu().detach() > 0.5) * 1)  # scikit-learn computations are numpy based;thus should run on CPU and without grads.
    train_acc_vec.append(train_accuracy)

    test_loss = test_loss / len(dl_test)
    test_loss_vec.append(test_loss)
    test_accuracy = accuracy_score(y_true_test.cpu(), (y_pred_test.cpu().detach() > 0.5) * 1)
    test_acc_vec.append(test_accuracy)

    print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
          test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')

plt.figure()
plt.plot(train_loss_vec, label='train')
plt.plot(test_loss_vec, label='test')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



