import _pickle
import contextlib
import io
import torch
import os
import numpy as np
from aux_eegproj_funcs_simplified import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchviz
import mne
# TODO change on every run! Be exhaustive
# run_desc = "changes: learning rate was .0001 before was getting a naive classifier" \
#            "         learning rate changed to .00005 " \
#            "loss types model architecture "
# dir_path = name_dir(run_desc)

# DATASET
table_name = 'sleep_1_10sec.csv'
eeg_ds = EEGDataset(EEGTransform, exp_table=build_experiment_tbl(table_name), filtered=0)
# eeg_ds = EEGDataset(EEGTransform, exp_table=build_simple_table())

# DATALOADERS
batch_sz = 1024 # TODO used to be 5. Was increased to make the loss and accuracy fluctuate less
# eeg_dl = DataLoader(eeg_ds, batch_size=batch_sz, shuffle=True, num_workers=0)
# dl_train, dl_val, dl_test = train_val_test_split_by_pid(dataset=eeg_ds,
#                                                         batch_size=batch_sz,
#                                                         train_rt=.6,
#                                                         val_rt=.2,  num_workers=0, verbose=1)

# TODO two arbitrary values for the loop to start
wF = 1
wM = 0
i = 0

while wF != wM and i != 100:  # TODO used to be without a loop. The loop makes sure that the training split is 50/50
    dl_train, dl_val, dl_test = tt_split_by_pid_mf(dataset=eeg_ds, batch_size=batch_sz, train_rt=.8,  num_workers=0, verbose=1)
    nF = sum(eeg_ds.table['sex (F=1)'][dl_train.sampler.indices] == 1)  # by segment
    nM = sum(eeg_ds.table['sex (F=1)'][dl_train.sampler.indices] == 2)
    wF = nM / (nF + nM)
    wM = nF / (nF + nM)
    i += 1
print('Took ', i, 'iterations to create a 50-50 male-female split')

# TODO add graphic information of the split
# use dl_.sample_indices to display the age and sex break down for each group tr tst val
# table of which patients are in which group
# makes a column where recordings have unique values
recs = (eeg_ds.table['subject']*2 + eeg_ds.table['night']).array
pts_by_group = pd.DataFrame()
groups = ('Train', 'Test', 'Validation')
# for dl_i, dl in enumerate((dl_train, dl_val, dl_test)):
#     if dl_i == 0:
#         pts_by_group = pd.DataFrame()
#     pts_dl = np.unique(eeg_ds.table['subject'][dl.sampler.indices])  # patients in group
#     pts_df = pd.DataFrame(pts_dl).columns = (groups[dl_i],)
#     pts_by_group = pd.concat((pts_by_group, pd.DataFrame(pts_dl)), axis=1)  # Table of which patients in which group
#     first_i_pt = [ind for ind in dl_val.sampler.indices
#                   if ind == np.where(eeg_ds.table['subject'] == eeg_ds.table['subject'][ind])[0][0]]
#     first_i_pt.sort()  # the indexes of the first occurrences of each unique pt in the dl
#     first_i_rec = [ind for ind in dl_val.sampler.indices if ind == np.where(recs == recs[ind])[0][0]]
#     first_i_rec.sort()  # the indexes of the first occurrences of each unique record in the dl
#     # Plot age hists
#     plt.figure()
#     plt.hist(eeg_ds.table['age'][dl.sampler.indices])  # by segment
#     plt.xlabel('age'), plt.ylabel('count by segment'), plt.title(groups[dl_i])
#     plt.show()
#     # Plot Sex bar plots
#     plt.figure()
#     nF = sum(eeg_ds.table['sex (F=1)'][dl.sampler.indices] == 1)  # by segment
#     nM = sum(eeg_ds.table['sex (F=1)'][dl.sampler.indices] == 2)
#     bars = plt.bar(x=(0, 1), height=(nF, nM))   # in table F1M2 in label F0M1
#     plt.bar_label(bars, labels=('F ({:.2f}%)'.format(100*nF/(nF+nM)), 'M ({:.2f}%)'.format(100*nM/(nF+nM))))
#     plt.xlabel('sex'), plt.ylabel('count by segment'), plt.title(groups[dl_i])
#     plt.show()
#     # Plot to show the representation of each pt in group; bar plot

# Display Data
# batch_to_show = 2
# plt.figure(figsize=(18, 6))
# for i, (signals, labels) in enumerate(dl_train):  # signals.shape torch.Size([4, 2, 60000])
#     labels = labels[0]
#     for j in range(batch_sz):
#         plt.subplot(batch_to_show, batch_sz, j+1+(i*batch_sz))
#         plt.imshow(signals[j][:,:200])
#         plt.title(labels[j].item())
#         plt.axis('off')
#     if i+1==batch_to_show:
#         break
# plt.show()


class my_weighted_bce():
    def __init__(self, weight: torch.tensor):
        self.w = weight

    def __call__(self, y_pred: torch.tensor, y_true: torch.tensor):
        # weight proportion of nMajority/nMin should be for dist in training only
        # the problem with the next line when using a GRU and a weighted BCE is that some predictions are 0 and log(0) is nan
        w_bce = torch.mean(-(torch.transpose(self.w, 0, 1)*(y_true*torch.log(y_pred))+(1-y_true)*torch.log(1-y_pred)))
        # depending on the size of self.w, you might want to use:
        # w_bce = torch.mean(-self.w*(y_true*torch.log(y_pred))+(1-y_true)*torch.log(1-y_pred)))
        return w_bce
        # for non-weighted-BCE-GRU use: return w_bce.item()


class GRUSexNet(nn.Module):
    def __init__(self, in_ax=1000, aa_channels=128, out_ya=1, num_layers=2, batch_first=True, bidirectional=True):
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
        self.FCs = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
            nn.Sigmoid())

        # def weights_init(m):
        #     if isinstance(m, nn.GRU):
        #         torch.nn.init.xavier_normal_(m.weight.data)
        #         torch.nn.init.xavier_normal_(m.bias.data)
        # self.apply(weights_init)

        # ------^^^^^^^^^------#

    def forward(self, x, a_in=None):
        # ------Your code------#
        # Set a condition to use only x as input if a_in=None.
        if a_in is None:
            GRU_output, a_out = self.GRU(x)
        else:
            GRU_output, a_out = self.GRU(x, a_in)
        y_pred = self.FCs(GRU_output)
        # ------^^^^^^^^^------#
        # features = self.CNN(x)
        # features = features.view(features.size(0), -1)  # reshape/flatten
        # scores = self.FCs(features)
        # # ------^^^^^^^^^------#
        # return torch.squeeze(scores)

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
            nn.Conv1d(in_channels=input_shape[1], out_channels=8, kernel_size=5, stride=1, padding=0, dilation=2),
            # TODO try changing the kernel sizes they were 3
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0, dilation=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0, dilation=2),
            nn.ReLU(),
            Residual(in_channels=32)
        )

        # Compute the CNN output size here to use as the input size for the fully-connected part.
        CNN_forward = self.CNN(torch.zeros(input_shape))

        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation (why?).
        self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 10),
            # input shape is the flattened CNN output, output shape is 100
            nn.ReLU(),
            nn.Linear(10, 1),  # We need 1 neuron as an output. input shape is 100 output shape is 1
            nn.Sigmoid()
        )

        # def weights_init(m):
        #     if isinstance(m, nn.Conv1d):
        #         torch.nn.init.xavier_normal_(m.weight.data)
        #         # torch.nn.init.xavier_normal_(m.bias.data)
        #         torch.nn.init.zeros_(m.bias)
        # self.apply(weights_init)

        # for param in self.parameters():
        #     print(param.data)
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

# sex_net = GRUSexNet(in_ax=signals.shape[2])
sex_net = eegSexNet(signals.shape) # TODO originally not commented
print('signal shape:')
print(signals.shape)
print(sex_net)


# F = 0; M = 1
# Optimizer for Sex sex_net
learning_rate = 0.001
opt_sex = torch.optim.Adam(params=sex_net.parameters(), lr=learning_rate)
# TODO I was just getting a naive classification. so switching to f1score for my loss func(?) can I
# NNLLoss and CROSSENTROPYLOSS have weights arguments
# idea to set the weight use the sampler indicies to slice into the tbl and get sex data
# subtract all by one (f0m1) sum and divide by length giving proproption of men use
# this as a corrective weight
# read up on diagnosing training curve trends


ce_w = nn.CrossEntropyLoss(weight=torch.tensor([wF, wM]))
bce = nn.BCELoss()
# bce = nn.BCELoss(weight=torch.transpose(torch.tensor([wF, wM]).repeat(5,1), 0, 1))  # good for binary classification # Loss Function for Sex eeg_sex_net
# bce = my_weighted_bce(weight=torch.tensor([wF, wM]).repeat(5,1))

n_epochs = 10
# Training SEXNET # placed into a function

print('Num of epochs is:', n_epochs)
print('Batch size is:', batch_sz)
print('Learning rate is:', learning_rate)
print('The table being used is:', table_name)


def Train_Sex_Net(epochs=n_epochs, fn='None', optimizer=opt_sex, loss_function=bce):
    global sex_net
    gpu_0 = torch.device('cpu')  # there are [0,128) cuda devices # TODO WHICH TO CHOOSE?
    label = 0  # select the sex label
    # print(gpu_0)
    train_loss_vec = []
    test_loss_vec = []
    val_loss_vec = []
    train_acc_vec = []
    test_acc_vec = []
    val_acc_vec = []
    for i_epoch in range(epochs):
        # print('prev_weights is:\n')
        # prev_weights = sex_net.parameters()
        # c0 = sex_net.CNN[0].weight
        # c2 = sex_net.CNN[2].weight
        # c4 = sex_net.CNN[4].weight
        # res = sex_net.CNN[6].weight
        # fc0 = sex_net.FCs[0].weight
        # fc2 = sex_net.FCs[2].weight
        # print(c0, '\n', c2, '\n', c4, '\n', res, '\n', fc0, '\n', fc2, '\n')

        train_loss = 0
        test_loss = 0
        val_loss = 0

        print(f'Epoch: {i_epoch + 1}/{epochs}')
        # Train set
        train_loss, y_true_train, y_pred_train = forward_epoch(sex_net, dl_train, loss_function, optimizer,
                                                                wM, train_loss,
                                                               to_train=True, desc='Train', device=gpu_0, label=label)
        # Test set
        test_loss, y_true_test, y_pred_test = forward_epoch(sex_net, dl_test, loss_function, optimizer, wM, test_loss,
                                                            to_train=False, desc='Test', device=gpu_0, label=label)
        # # Validation set
        # val_loss, y_true_val, y_pred_val = forward_epoch(sex_net, dl_val, loss_function, optimizer, val_loss,
        #                                                  to_train=False, desc='Validation', device=gpu_0, label=label, weight=wM)

        # Metrics:
        train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
        test_loss = test_loss / len(dl_test)
        # val_loss = val_loss / len(dl_val)
        train_loss_vec.append(train_loss)
        test_loss_vec.append(test_loss)
        # val_loss_vec.append(val_loss)

        # scikit-learn computations are numpy based;thus should run on CPU and without grads.
        train_accuracy = accuracy_score(y_true_train.cpu(),
                                        (y_pred_train.cpu().detach() > 0.5) * 1)
        test_accuracy = accuracy_score(y_true_test.cpu(),
                                       (y_pred_test.cpu().detach() > 0.5) * 1)
        # val_accuracy = accuracy_score(y_true_val.cpu(),
        #                               (y_pred_val.cpu().detach() > 0.5) * 1)
        train_acc_vec.append(train_accuracy)
        test_acc_vec.append(test_accuracy)
        # val_acc_vec.append(val_accuracy)

        # new_weights = sex_net.parameters()
        # print('new_weights - prev_weights is:\n')
        # print(1000*(c0 - sex_net.CNN[0].weight))
        # print(1000*(c2 - sex_net.CNN[2].weight))
        # print(1000*(c4 - sex_net.CNN[4].weight))
        # print(1000*(res - sex_net.CNN[6].weight))
        # print(1000*(fc0 - sex_net.FCs[0].weight))
        # print(1000*(fc2 - sex_net.FCs[2].weight))
        #
        # print('torch.sum(new_weights - prev_weights) is:\n')
        # print(1000*torch.std(c0 - sex_net.CNN[0].weight))
        # print(1000*torch.std(c2 - sex_net.CNN[2].weight))
        # print(1000*torch.std(c4 - sex_net.CNN[4].weight))
        # print(1000*torch.std(res - sex_net.CNN[6].weight))
        # print(1000*torch.std(fc0 - sex_net.FCs[0].weight))
        # print(1000*torch.std(fc2 - sex_net.FCs[2].weight))

        print('\n')
        print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
              test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')
    try:
        if fn != 'None':
            if fn[-7:] != ".pickle":
                fn = fn + ".pickle"
            torch.save(sex_net.state_dict(), fn)
            torch.save(opt_sex.state_dict(), fn[:-7]+'_opt'+fn[-7:])
            #torch.save(sex_net, f=fn)
            print('saved model')
    except:
        print("didn't save")
        pass
    return (train_loss_vec, train_acc_vec), (test_loss_vec, test_acc_vec) #, (val_loss_vec, val_acc_vec)


f0 = 'sex_k7'  # file format '.pickle' added automatically
train_res, test_res = Train_Sex_Net(epochs=n_epochs, fn=f0)
# SAVING AND LOADING MODEL/OPTIMIZER reference
# torch.save(sex_net.state_dict(), 'sex_net_numero_uno.pickle')
# torch.save(opt_sex.state_dict(), 'sex_net_numero_uno_opt.pickle')
# sex_net_loaded = eegSexNet(signals.shape)
# sex_net_loaded.load_state_dict(torch.load('sex_net_numero_uno.pickle'))
# sex_net_loaded.eval()


# PLOT SOME RESULTS FROM TRAINING SEXNET
plt.figure()
plt.plot(train_res[0], label='train')
plt.plot(test_res[0], label='test')
plt.title('SexNet training')
plt.title('sex net training')
plt.ylabel('BCE loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Plot confusion Matrices for the Sex net
# perform on train, test and validation
# gpu_0 = torch.device(2)
gpu_0=torch.device('cpu')
_, y_true_sex_train, y_pred_sex_train = forward_epoch(sex_net, dl_train, bce, opt_sex,
                                                      to_train=False, desc='Train',
                                                      device=gpu_0, weight=torch.tensor([wF, wM]))  # run on the validation data
y_pred_sex_train = (y_pred_sex_train.cpu().detach() > 0.5)*1
y_true_sex_train = y_true_sex_train.cpu()
ConfusionMatrixDisplay.from_predictions(y_true_sex_train, y_pred_sex_train, cmap='PuRd')  # TODO title(?)
plt.title('Train CM')
plt.show()
_, y_true_sex_test, y_pred_sex_test = forward_epoch(sex_net, dl_test, bce, opt_sex,
                                                    to_train=False,
                                                    desc='Test', device=gpu_0, weight=torch.tensor([wF, wM]))  # run on the validation data
y_pred_sex_val = (y_pred_sex_test.cpu().detach() > 0.5)*1
y_true_sex_val = y_true_sex_test.cpu()
ConfusionMatrixDisplay.from_predictions(y_true_sex_val, y_pred_sex_val, cmap='PuRd')
plt.title('Test CM')
plt.show()
_, y_true_sex_val, y_pred_sex_val = forward_epoch(sex_net, dl_val, bce, opt_sex,
                                                  to_train=False,
                                                  desc='Validation', device=gpu_0, weight=torch.tensor([wF, wM])) # run on the validation data
y_pred_sex_val = (y_pred_sex_val.cpu().detach() > 0.5)*1
y_true_sex_val = y_true_sex_val.cpu()
ConfusionMatrixDisplay.from_predictions(y_true_sex_val, y_pred_sex_val, cmap='PuRd')
plt.title('Validation CM')
plt.show()

test_f1_score = f1_score(y_true_sex_val, y_pred_sex_val)  # do more with this score
print(f'test_f1-score={round(test_f1_score, 3)}')


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
            nn.Conv1d(in_channels=input_shape[1], out_channels=8, kernel_size=7, stride=1, padding=0, dilation=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=64, kernel_size=7, stride=1, padding=0, dilation=3),
            nn.ReLU(),
            Residual(in_channels=16)
        )

        # Compute the CNN output size here to use as the input size for the fully-connected part.
        CNN_forward = self.CNN(torch.zeros(input_shape))

        # Define the fully-connected layers in a nn.Sequential.
        # Use nn.Linear for a fully-connected layer.
        # Use nn.Sigmoid as the final activation (why?).
        self.FCs = nn.Sequential(
            nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 10),
            # input shape is the flattened CNN output, output shape is 100
            nn.ReLU(),
            nn.Linear(10, 1),  # We need 1 neuron as an output. input shape is 100 output shape is 1
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


learning_rate = 0.0001
age_net = eegAgeNet(signals.shape)
opt_age = torch.optim.Adam(params=age_net.parameters(), lr=learning_rate)
mse = nn.MSELoss()  # good for regression

# the loss in the next function should be changed
def Train_Age_Net(epochs=n_epochs, fn='None', optimizer=opt_age, loss_function=mse):
    global age_net
    # gpu_0 = torch.device(2)  # there are [0,128) cuda devices
    gpu_0=torch.device('cpu')
    label = 1  # select the age label
    # print(gpu_0)
    train_loss_vec = []
    test_loss_vec = []
    val_loss_vec = []
    train_acc_vec = []
    test_acc_vec = []
    val_acc_vec = []
    for i_epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        val_loss = 0

        print(f'Epoch: {i_epoch + 1}/{epochs}')
        # Train set
        train_loss, y_true_train, y_pred_train = forward_epoch(age_net, dl_train, loss_function, optimizer, train_loss,
                                                               to_train=True, desc='Train', device=gpu_0, label=label)
        # Test set
        test_loss, y_true_test, y_pred_test = forward_epoch(age_net, dl_test, loss_function, optimizer, test_loss,
                                                            to_train=False, desc='Test', device=gpu_0, label=label)
        # Validation set
        # val_loss, y_true_val, y_pred_val = forward_epoch(age_net, dl_val, loss_function, optimizer, val_loss,
        #                                                  to_train=False, desc='Test', device=gpu_0, label=label)

        # Metrics:
        train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
        test_loss = test_loss / len(dl_test)
        # val_loss = val_loss / len(dl_val)
        train_loss_vec.append(train_loss)
        test_loss_vec.append(test_loss)
        # val_loss_vec.append(val_loss)

        # scikit-learn computations are numpy based;thus should run on CPU and without grads.
        train_accuracy = accuracy_score(y_true_train.cpu(),
                                        (y_pred_train.cpu().detach() > 0.5) * 1)
        test_accuracy = accuracy_score(y_true_test.cpu(),
                                       (y_pred_test.cpu().detach() > 0.5) * 1)
        # val_accuracy = accuracy_score(y_true_val.cpu(),
        #                               (y_pred_val.cpu().detach() > 0.5) * 1)
        train_acc_vec.append(train_accuracy)
        test_acc_vec.append(test_accuracy)
        # val_acc_vec.append(val_accuracy)

        print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
              test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')

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
    return (train_loss_vec, train_acc_vec), (test_loss_vec, test_acc_vec)  # (val_loss_vec, val_acc_vec)


n_epochs = 60
f1 = 'age_k7'
train_res, test_res = Train_Age_Net(epochs=n_epochs, fn=f1)

plt.figure()
plt.plot(train_res[0], label='train')  # train_loss_vec
plt.plot(test_res[0], label='test')  # test_loss_vec
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.title('Age Net Training Loss')
plt.legend()
plt.show()
# TODO print a communicative graphic of the results. scatter plot x actual age y predicted age
# TODO analysis of prediction. is it just the average age. is it better in a specific age range
# TODO show the ages and sex distribution of the split data


