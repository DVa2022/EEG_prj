import contextlib
import io
import os
import numpy as np
import torch as torch
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
# Hyperparameters
old_dir_path = '/home/stu16/EEG_proj/runs/368/'
exp_tbl_fn = 'sleep_34_10sec.csv'
batch_sz = 256
num_workers = 8
n_epochs_sex = 25
n_epochs_age = 25
learning_rate_sex = 1e-04 * 33
learning_rate_age = 1e-04 * 1
w_decay_s = 0.0005  # 1e-2  # l2

# DATASET
eeg_ds = EEGDataset(EEGTransform, exp_table=build_experiment_tbl(exp_tbl_fn))
# DATALOADERS
dl_train, dl_test, dl_val = recover_dls(eeg_ds, old_dir_path, batch_size=batch_sz, num_workers=num_workers)
groups = ('Train', 'Test', 'Validation')
# when ready for final eval
olen_dl_train, olen_dl_test, olen_dl_val = len(dl_train.sampler.indices), len(dl_test.sampler.indices), \
                                           len(dl_val.sampler.indices)  # original dl length
signals = dataviz(batch_sz, dl_train, show=False)
ss = signals.size()
del signals
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
sex_net = eegSexNetblstm(ss)
sex_net = sex_net.double()
opt_sex = torch.optim.Adam(params=sex_net.parameters(), lr=learning_rate_sex, weight_decay=w_decay_s)
# Load the optimizer and model for sexnet
with open(f'sex_model.pickle', 'rb') as f:
    sex_net.load_state_dict(torch.load(f))
with open(f'sex_model_opt.pickle', 'rb') as f:
    opt_sex.load_state_dict(torch.load(f))

try:
    dls = (dl_train, dl_test, dl_val)
except NameError:
    dls = (dl_train, dl_test)
dl_all_ordered = DataLoader(eeg_ds, batch_size=batch_sz, sampler=SequentialSampler(range(len(eeg_ds.table)+batch_sz-(len(eeg_ds.table) % batch_sz))),
                            num_workers=num_workers, drop_last=True)  # need to have these in order hence Seq Sampler # get a complete final batch by adding extra indices to seq # TODO work code to ignore these

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


osmps_dl_train, osmps_dl_test, osmpd_dl_val = dl_train.sampler.indices, dl_test.sampler.indices, \
                                              dl_val.sampler.indices  # original sampke indices in dls
del dl_train, dl_test, dl_val

# Evaluate network: on all the data (no training): use dataloader samplers to separate data by group later
_, y_true_sex, y_pred_sex = forward_epoch(sex_net, dl_all_ordered, bcel, opt_sex, # since this uses dl_all_ordered
                                          to_train=False, desc='all',             # preds share indices with samples in
                                          device=gpu_0)                           # exp table
y_pred_sex = y_pred_sex.cpu().detach()                # probability prediction decimal btw 0-1
y_pred_sex = y_pred_sex[0:len(eeg_ds.table)]  # drop the values used only to fill out the final batch
y_true_sex = y_true_sex.cpu().detach()                # label of sample
y_true_sex = y_true_sex[0:len(eeg_ds.table)]  # drop the values used only to fill out the final batch
# functions to return the results by group [rec as in simple table] [mean] [modes]
rec_preds_p, rec_preds_c, rec_true = sort_pred_by_rec(eeg_ds, y_pred_sex, y_true_sex, s1a0=1)  # still need this if not dl_all
res_tr, res_tst, res_val = sort_pred_by_dl(eeg_ds, dls, rec_preds_p, rec_preds_c)
res_sex = res_tr, res_tst, res_val
# recs_dltr, means_tr, modes_tr, true_tr = res


for i, dl in enumerate(dls):    # CM by the sample
    cm_samp = ConfusionMatrixDisplay.from_predictions(y_true_sex[dl.sampler.indices], (y_pred_sex[dl.sampler.indices] > 0.5) * 1, cmap='PuRd')
    plt.title('Sex {} CM by sample'.format(groups[i]))
    plt.show()
for i, res in enumerate(res_sex):  # CM by the rec mean
    cm_mean = ConfusionMatrixDisplay.from_predictions((np.array(res[1]) > .5)*1, res[3], cmap='PuRd')
    plt.title('Sex {} CM mean by pt rec'.format(groups[i]))
    plt.show()
for i, res in enumerate(res_sex):  # CM by the rec mode
    cm_mode = ConfusionMatrixDisplay.from_predictions(res[2], res[3], cmap='PuRd')
    plt.title('Sex {} CM mode by pt rec'.format(groups[i]))
    plt.show()

    # if dir_path:
    #     plt.savefig(dir_path+"{}CM".format(groups[i]) + ".png")
    # plt.show()
for i, dl in enumerate(dls):  # f1 score on mean mode
    f1_score_samp = f1_score(y_true_sex[dl.sampler.indices], (y_pred_sex[dl.sampler.indices] > 0.5) * 1)
    f1_score_mean = f1_score(np.array(res_sex[i][3])*1., (np.array(res_sex[i][1]) > 0.5) * 1)
    f1_score_mode = f1_score(np.array(res_sex[i][3])*1., np.array(res_sex[i][2]))
    f1_report_samp = 'sex {group} samp f1-score = {score}'.format(group=groups[i], score=round(f1_score_samp, 3))
    f1_report_mean = 'sex {group} mean f1-score = {score}'.format(group=groups[i], score=round(f1_score_mean, 3))
    f1_report_mode = 'sex {group} mode f1-score = {score}'.format(group=groups[i], score=round(f1_score_mode, 3))
    print(f1_report_samp)
    print(f1_report_mean)
    print(f1_report_mode)

del opt_sex, sex_net,
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


age_net = eegAgeNet(signals.shape)
age_net = age_net.double()
opt_age = torch.optim.Adam(params=age_net.parameters(), lr=learning_rate_age)

# TODO
# Load the optimizer and model for sexnet
with open(f'age.pickle', 'rb') as f:
    age_net.load_state_dict(torch.load(f))
with open(f'age_opt.pickle', 'rb') as f:
    opt_age.load_state_dict(torch.load(f))

mse = nn.MSELoss()  # good for regression
sm_l1 = nn.SmoothL1Loss()  # beta = 1 by default

# Evaluate network: on all the data (no training): use dataloader samplers to separate data by group later
_, y_true_age, y_pred_age = forward_epoch(age_net, dl_all_ordered, mse, opt_age, # since this uses dl_all_ordered
                                          to_train=False, desc='all',             # preds share indices with samples in
                                          device=gpu_0, label=1)                           # exp table
y_pred_age = y_pred_age.cpu().detach()                # probability prediction decimal btw 0-1
y_pred_age = y_pred_age[0:len(eeg_ds.table)]  # drop the values used only to fill out the final batch
y_true_age = y_true_age.cpu().detach()                # label of sample
y_true_age = y_true_age[0:len(eeg_ds.table)]  # drop the values used only to fill out the final batch
# function to return the results by group [rec as in simple table] [mean] [modes]
rec_preds_p, rec_preds_c, rec_true = sort_pred_by_rec(eeg_ds, y_pred_age, y_true_age, s1a0=0)
res_tr, res_tst, res_val = sort_pred_by_dl(eeg_ds, dls, rec_preds_p, rec_preds_c, s1a0=0)
# recs_dltr, means_tr, modes_tr, true_tr = res  # modes are meaningless now for age
res_age = res_tr, res_tst, res_val

# Scatter plot of ages by sample
plt.figure()
plt.scatter(y_true_age[dl_train.sampler.indices].numpy(), y_pred_age[dl_train.sampler.indices].numpy(), label='train')
plt.scatter(y_true_age[dl_test.sampler.indices].numpy(), y_pred_age[dl_test.sampler.indices].numpy(), label='test')
plt.scatter(y_true_age[dl_val.sampler.indices].numpy(), y_pred_age[dl_val.sampler.indices].numpy(), label='val')
plt.ylabel('Pred. age')
plt.xlabel('True age')
plt.title('Age Predictions by sample')
plt.legend()

# Scatter plot of ages by rec
plt.figure()
plt.scatter(res_age[0][3], res_age[0][1], label='train')
plt.scatter(res_age[1][3], res_age[1][1], label='test')
plt.scatter(res_age[2][3], res_age[2][1], label='val')
plt.ylabel('Pred. age')
plt.xlabel('True age')
plt.title('Age Predictions by rec')
plt.legend()
if old_dir_path:
    plt.savefig(old_dir_path+"age_predictions_rec" + ".png")
plt.show()

# TODO analysis of prediction. is it just the average age. is it better in a specific age range



