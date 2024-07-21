import contextlib
import io
import os
import numpy as np
from aux_eeg_rewrite import *
from aux_eeg_rewrite import LSTM_adapted as aLSTM
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchviz
import mne
import pickle
global gpu_0
gpu_0 = torch.device(2)
# TODO change on every run! Be exhaustive
# TODO batch size is a good hyperparameter,
# Hyperparameters
exp_tbl_fn = 'sleep_34_10sec.csv'
batch_sz = 256
num_workers = 8
n_epochs_sex = 35
n_epochs_age = 35
learning_rate_sex = 1e-05
learning_rate_age = 1e-05
w_decay_s = 0.005  # 1e-2  # l2

run_desc = "BLSTM var   ATV " \
           "\nExp table file: {fn}" \
            "\nBatch size: {bs}" \
            "\nNum workers: {nw}" \
            "\nn_epochs_sex: {nes}" \
            "\nn_epochs_age: {nea}" \
            "\nlearning_rate_sex: {lrs}" \
            "\nlearning_rate_age: {lra}" \
            "\nweight decay/ L2 sex:{wds}\n".format(fn=exp_tbl_fn, bs=batch_sz, nw=num_workers, nes=n_epochs_sex, nea=n_epochs_age,
                                                    lrs=learning_rate_sex, lra=learning_rate_age, wds=w_decay_s)
# how do I get the net after desired # of epochs without saving after each iteration
dir_path = name_dir(run_desc)
if dir_path:
    print('Run # ', dir_path.split('/')[-2])
    print(run_desc)

# DATASET
# eeg_ds = EEGDataset(EEGTransform, exp_table=build_experiment_tbl(exp_tbl_fn))
eeg_ds = EEGDataset_variable_creation(EEGTransform, exp_table=build_experiment_tbl(exp_tbl_fn), table_name=exp_tbl_fn, filtered=0)

append_desc('\nexp_tbl_fn: '+exp_tbl_fn+'/n', dir_path)

# DATALOADERS
dl_train,  dl_test, dl_val = train_val_test_split_by_pid(eeg_ds, batch_sz, train_rt=.6, val_rt=.2,  num_workers=num_workers, verbose=1)
groups = ('Train', 'Test', 'Validation')
# reduce the size by a factor of 1/3
dl_train.sampler.indices = dl_train.sampler.indices[:3*len(dl_train.sampler.indices)//4]
dl_test.sampler.indices = dl_test.sampler.indices[:3*len(dl_test.sampler.indices)//4]
dl_val.sampler.indices = dl_val.sampler.indices[:3*len(dl_val.sampler.indices)//4]

try:
    dls = (dl_train, dl_test, dl_val)
except NameError:
    dls = (dl_train, dl_test)

# Save the data loaders
if dir_path:
    for dl_i, dl in enumerate(dls):
        with open(dir_path + "dl_" + groups[dl_i] + ".csv", "wb") as f:
            DF = pd.DataFrame(dl.sampler.indices)
            DF.to_csv(f)

# Split visualization: Table of patients by group.
pt_split_tbl(eeg_ds, dls, groups, dir_path)

# Plot age hists
age_split_hist(eeg_ds, dls, groups, dir_path, show=False)

# Plot Sex split bar plots (by segment)
sex_split_bar(eeg_ds, dls, groups, dir_path, show=False)

# Display Data
signals = dataviz(batch_sz, dl_train, show=False)

# Define a network to classify Sex:
# Basing network off model seen in literature
# Architecture of the proposed Deep BLSTM_LSTM model, with one BLSTM
# and two LSTM layers followed by dropout, batch normalization and dense layers
class eegSexNetblstm(nn.Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
                                               (batch_sz,n_channels, seq_length)
        """
        super().__init__()
        # ------Your code------#
        self.BLSTM = nn.Sequential(
            # BLSTM with dropout 0.2. Use # sequence length as the first layer input size.
            aLSTM(input_size=input_shape[2], hidden_size=256, batch_first=True, num_layers=2, dropout=0.2,
                  bidirectional=True),
            # Batch normalization
            nn.BatchNorm1d(num_features=input_shape[1]),
            aLSTM(input_size=256*2, hidden_size=128, batch_first=True, bidirectional=False),
            # Batch normalization
            nn.BatchNorm1d(num_features=input_shape[1]),
            aLSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=False),
            # Batch normalization
            nn.BatchNorm1d(num_features=input_shape[1])
        )
        # Compute the CNN output size here to use as the input size for the fully-connected part.
        self.BLSTM = self.BLSTM.double()
        BLSTM_forward = self.BLSTM(torch.zeros(input_shape).double())
        # Define the fully-connected layers in a nn.Sequential.
        #       Use nn.Linear for a fully-connected layer.
        #       Use nn.Sigmoid as the final activation.
        self.FCs = nn.Sequential(
            nn.Linear(BLSTM_forward.shape[1] * BLSTM_forward.shape[2], 32),
            # input shape is the flattened LSTM output, output shape is 32
            nn.ReLU(),
            nn.Linear(32, 1),  # We need 1 neuron as an output. input shape is 100 output shape is 1
            nn.Sigmoid()
        )
        # ------^^^^^^^^^------#

    def forward(self, x):
        # ------Your code------#
        # Forward through the CNN by passing x, flatten and then forward through the linears.
        features = self.BLSTM(x.double())
        # print("ft size1: ", features.shape)
        features = features.view(features.size(0), -1)  # reshape/flatten
        # print("ft size2: ", features.shape)
        scores = self.FCs(features)
        # print('scores', scores, ' shape ', scores.shape)
        # ------^^^^^^^^^------#
        return torch.squeeze(scores)

# Instantiate the network
sex_net = eegSexNetblstm(signals.size())
sex_net = sex_net.double()
print('signal shape: ', signals.shape)
print(sex_net)

# Optimizer for Sex sex_net  F = 0; M = 1
opt_sex = torch.optim.Adam(params=sex_net.parameters(), lr=learning_rate_sex, weight_decay=w_decay_s, maximize=0)
append_desc('\n SexNet optimizer: {} '.format(opt_sex.__repr__), dir_path)
append_desc('\n SexNet architecture: \n {} '.format(sex_net.__repr__), dir_path)
# TODO I was just getting a naive classification. so switching to f1score for my loss func(?) can I
# NNLLoss and CROSSENTROPYLOSS have weights arguments
# idea to set the weight use the sampler indicies to slice into the tbl and get sex data
# subtract all by one (f0m1) sum and divide by length giving proproption of men use
# this as a corrective weight
# read up on diagnosing training curve trends
# Corrective weights for loss
nF = sum(eeg_ds.table['sex (F=1)'][dl_train.sampler.indices] == 1)  # by segment
nM = sum(eeg_ds.table['sex (F=1)'][dl_train.sampler.indices] == 2)  # wF = nM/(nF+nM) # wM = nF/(nF+nM)
nFtst = sum(eeg_ds.table['sex (F=1)'][dl_test.sampler.indices] == 1)
nMtst = sum(eeg_ds.table['sex (F=1)'][dl_test.sampler.indices] == 2)
pos_weight = torch.tensor([nF/nM]*batch_sz, device=gpu_0)  # must be a torch to work even for bcel
print('nF in train: ', nF, ' nM in train: ', nM, 'positive weight: ', nF/nM, ' %F: ', nF/(nF+nM))
print('nF in test: ', nFtst, ' nM in test: ', nMtst, ' %F: ', nFtst/(nFtst+nMtst))
# Loss Function
bcel = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# bce = nn.BCELoss(weight=pos_weight)  # good for binary classification # Loss Function for Sex eeg_sex_net
mbce = my_weighted_bce(pos_weight)

# Training SEXNET # placed into a function
def Train_Sex_Net(epochs=10, fn='None', optimizer=opt_sex, loss_function=bcel):
    global sex_net
    global gpu_0
    label = 0  # select the sex label
    train_loss_vec = []
    test_loss_vec = []
    train_acc_vec = []
    test_acc_vec = []
    for i_epoch in range(epochs):
        print(f'Epoch: {i_epoch + 1}/{epochs}')
        train_loss = 0
        test_loss = 0
        # Train set
        train_loss, y_true_train, y_pred_train = forward_epoch(sex_net, dl_train, loss_function, optimizer,
                                                               train_loss, to_train=True, desc='Train',
                                                               device=gpu_0, label=label)
        # Test set
        test_loss, y_true_test, y_pred_test = forward_epoch(sex_net, dl_test, loss_function, optimizer, test_loss,
                                                            to_train=False, desc='Test', device=gpu_0, label=label)
        # Metrics:
        train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
        test_loss = test_loss / len(dl_test)     # (not fully accurate b/c of drop last but ok)
        train_loss_vec.append(train_loss)
        test_loss_vec.append(test_loss)

        # scikit-learn computations are numpy based;thus should run on CPU and without grads.
        train_accuracy = accuracy_score(y_true_train.cpu(),
                                        (y_pred_train.cpu().detach() > 0.5) * 1)
        test_accuracy = accuracy_score(y_true_test.cpu(),
                                       (y_pred_test.cpu().detach() > 0.5) * 1)
        train_acc_vec.append(train_accuracy)
        test_acc_vec.append(test_accuracy)

        print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
              test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')
    try:
        if fn != 'None':
            if fn[-7:] != ".pickle":
                fn = fn + ".pickle"
            torch.save(sex_net.state_dict(), fn)
            torch.save(opt_sex.state_dict(), fn[:-7]+'_opt'+fn[-7:])
            print('saved model')
    except:
        print("didn't save")
        pass
    return (train_loss_vec, train_acc_vec), (test_loss_vec, test_acc_vec)

fn_sex_model = dir_path + 'sex_model' if dir_path else 'None'  # file format '.pickle' added automatically
train_res, val_res = Train_Sex_Net(epochs=n_epochs_sex, fn=fn_sex_model)


# PLOT SOME RESULTS lossFROM TRAINING SEXNET
plt.figure()
plt.plot(train_res[0], label='train')
plt.plot(val_res[0], label='test')
plt.title('SexNet training')
plt.ylabel('BCE loss')
plt.xlabel('Epoch')
plt.legend()
if dir_path:
    plt.savefig(dir_path+"SEXNET_training_loss" + ".png")
plt.show()

# PLOT SOME RESULTS accuracy FROM TRAINING SEXNET
plt.figure()
plt.plot(train_res[1], label='train')
plt.plot(val_res[1], label='test')
plt.title('SexNet training accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend()
if dir_path:
    plt.savefig(dir_path+"SEXNET_training_accuracy" + ".png")
plt.show()

ss = signals.shape
del sex_net, dls, opt_sex, train_res, val_res, signals


# Define AGE network:
class eegAgeNet(nn.Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input tensor shape - every batch size will be ok as it is used to compute the FCs input size.
        """
        super().__init__()
        # ------Your code------#
        self.BLSTM = nn.Sequential(
            # BLSTM with dropout 0.2. Use # sequence length as the first layer input size.
            aLSTM(input_size=input_shape[2], hidden_size=256, batch_first=True, num_layers=2, dropout=0.2,
                  bidirectional=True),
            # Batch normalization
            nn.BatchNorm1d(num_features=input_shape[1]),
            aLSTM(input_size=256 * 2, hidden_size=128, batch_first=True, bidirectional=False),
            # Batch normalization
            nn.BatchNorm1d(num_features=input_shape[1]),
            aLSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=False),
            # Batch normalization
            nn.BatchNorm1d(num_features=input_shape[1])
        )
        # Compute the BLSTM output size here to use as the input size for the fully-connected part.
        self.BLSTM = self.BLSTM.double()
        BLSTM_forward = self.BLSTM(torch.zeros(input_shape).double())

        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation (why?).
        self.FCs = nn.Sequential(
            nn.Linear(BLSTM_forward.shape[1] * BLSTM_forward.shape[2], 100),
            # input shape is the flattened BLSTM output, output shape is 100
            nn.ReLU(),
            nn.Linear(100, 1),  # We need 1 neuron as an output. input shape is 100 output shape is 1
            nn.ReLU(),
        )
        # ------^^^^^^^^^------#

    def forward(self, x):
        # ------Your code------#
        # Forward through the CNN by passing x, flatten and then forward through the linears.
        features = self.BLSTM(x.double())
        features = features.view(features.size(0), -1)  # reshape/flatten
        scores = self.FCs(features)
        # ------^^^^^^^^^------#
        return torch.squeeze(scores)


age_net = eegAgeNet(ss)
opt_age = torch.optim.Adam(params=age_net.parameters(), lr=learning_rate_age)
mse = nn.MSELoss()  # good for regression
sm_l1 = nn.SmoothL1Loss()  # beta = 1 by default
append_desc('\n AGE Net optimizer: {} \n'.format(opt_age.__repr__), dir_path)
append_desc('\n Age Net architecture: \n '+age_net.__repr__(), dir_path)


def Train_Age_Net(epochs=n_epochs_age, fn='None', optimizer=opt_age, loss_function=mse):
    global age_net
    global gpu_0
    label = 1  # select the age label
    train_loss_vec = []
    test_loss_vec = []
    for i_epoch in range(epochs):
        print(f'Epoch: {i_epoch + 1}/{epochs}')
        train_loss = 0
        test_loss = 0
        # Train set
        train_loss, y_true_train, y_pred_train = forward_epoch(age_net, dl_train, loss_function, optimizer, train_loss,
                                                               to_train=True, desc='Train', device=gpu_0, label=label)
        # Test set
        test_loss, y_true_test, y_pred_test = forward_epoch(age_net, dl_test, loss_function, optimizer, test_loss,
                                                            to_train=False, desc='Test', device=gpu_0, label=label)
        # Metrics:
        train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
        test_loss = test_loss / len(dl_test)
        train_loss_vec.append(train_loss)
        test_loss_vec.append(test_loss)
        print(f'train_loss={round(train_loss, 3)} \
              test_loss={round(test_loss, 3)}')
    try:
        if fn != 'None':
            if fn[-7:] != ".pickle":
                fn = fn + ".pickle"
            torch.save(age_net.state_dict(), fn)
            torch.save(opt_age.state_dict(), fn[:-7]+'_opt'+fn[-7:])
            print('saved agenet after training')
    except:
        print("didn't save agenet after training")
        pass
    return (train_loss_vec, y_true_train, y_pred_train), \
           (test_loss_vec, y_true_test, y_pred_test)


fn_age_model = dir_path + 'age' if dir_path else 'None'
train_res, test_res = Train_Age_Net(epochs=n_epochs_age, fn=fn_age_model)
# _loss_vev, y_true_, y_pred_ = _res

plt.figure()  # Age Net Training Loss
plt.plot(train_res[0], label='train')  # train_loss_vec
plt.plot(test_res[0], label='test')  # test_loss_vec
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.title('Age Net Training Loss')
plt.legend()
if dir_path:
    plt.savefig(dir_path+"AGENET_training" + ".png")
plt.show()

# Scatter plot of ages by sample
plt.figure()
plt.scatter(train_res[1].cpu().detach() , train_res[2].cpu().detach() , label='train')
plt.scatter(test_res[1].cpu().detach() , test_res[2].cpu().detach() , label='test')
plt.ylabel('Pred. age')
plt.xlabel('True age')
plt.title('Age Predictions by sample')
plt.legend()
if dir_path:
    plt.savefig(dir_path+"age_predictions_rec" + ".png")
plt.show()

# TODO analysis of prediction. is it just the average age. is it better in a specific age range

