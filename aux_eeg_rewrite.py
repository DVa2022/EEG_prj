import os
import pandas as pd
import contextlib
import io
import datetime
import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

def build_simple_table():
    """
    Builds "simple table.csv"
    """
    # if output is saved does not rerun
    try:
        simple_tbl = pd.read_csv("simple table.csv", header=0, index_col=0)
        print("{} Already exists".format("simple table.csv"))
        return simple_tbl
    except:
        print("{} did not already exist".format("simple table.csv"))
        print("creating now")

    if 'C:\\Users\\atver' in os.getcwd():  # is my cpu or is not ie triton
        table_paths = ("C:\\Users\\atver\\mne_data\\physionet-sleep-data\\SC-subjects.xls","C:\\Users\\atver\\mne_data\\physionet-sleep-data\\ST-subjects.xls")
    else: # on triton
        table_paths = ("\\home\\stu16\\physionet_data\\SC-subjects.xls", "\\home\\stu16\\physionet_data\\ST-subjects.xls")
    # Build simple_table ###
    sctable_path, sttable_path = table_paths
    sctable = pd.read_excel(sctable_path, sheet_name=0, header=0)
    sttable = pd.read_excel(sttable_path, sheet_name=0, header=1) # has two lines of header
    # combine both table data into one panda SC then below ST placebo then ST temazepan
    # using column orginzation from SC
    sub = sttable['Nr'] + 82  # increase index so that ST patient index follows SC index
    nightp = sttable['night nr']
    age = sttable['Age']
    sex = (- (sttable['M1/F2']-1)) + 2  # match convention in SC F=1 M=2
    offp = sttable['lights off']
    nightt = sttable['night nr.1']
    offt = sttable['lights off.1']
    night1 = pd.concat([sub, nightp, age, sex, offp], axis=1)
    night1.columns = sctable.columns
    night2 = pd.concat([sub, nightt, age, sex, offt], axis=1)
    night2.columns = sctable.columns
    # ignore_index stops the row index from restarting
    simple_table = pd.concat([sctable, night1, night2], ignore_index=True)
    # save the built table
    simple_table.to_csv("simple table.csv")
    return simple_table  # table of segments


class PathFinder:
    def __init__(self, filtered, tbl):
        # filtered: a flag to choose either original data (False) or filtered data (True)
        # tbl: the table to use for looking up the recordings. uses the index in the
        #      table and the corresponding pid and night that appear there pd.df
        self.tbl = tbl
        # set the path to the data
        self.f = filtered
        if self.f:
            if 'C:\\Users\\atver' in os.getcwd():  # is my cpu
                self.DB_path = "C:/Users/atver/mne_data/filt_data/"  # getting data from filtered data
            elif 'David' in os.getcwd():
                self.DB_path = 'C:/Users/David/mne_data/filt_data/'
            else:  # or is not ie triton
                self.DB_path = "/home/stu16/filt_data/"
        else:  # original data
            if 'C:\\Users\\atver' in os.getcwd():  # is my cpu
                self.DB_path = "C:/Users/atver/mne_data/physionet-sleep-data/"  # getting data from unfiltered data
            elif 'David' in os.getcwd():
                self.DB_path = 'C:/Users/David/mne_data/physionet-sleep-data/'
            else:  # or is not ie triton
                self.DB_path = "/home/stu16/physionet_data/"

    def get_paths(self, index: int):
        # the function gets the paths of the recording [0] for leads [1] for annotations
        # of either the filtered files or the original ones on triton and my machine
        # index: index of recording as appearing in the experiment table or simple_table
        db = self.DB_path

        ss = self.tbl['subject'][index]  # subject and night of sample at index in
        n = self.tbl['night'][index]     # experiment table
        if ss < 83:  # part of SC
            pid = int(ss)
            # SC4ssNEO-PSG.edf where ss is the subject number, and N is the night
            rec_title = 'SC4{:02d}{}'.format(pid, n)
            if self.f:
                leads = [x for x in os.listdir(db) if ((rec_title in x) and ('raw' in x))]
            else:
                leads = [x for x in os.listdir(db) if ((rec_title in x) and ('PSG' in x))]
            assert (len(leads) == 1)
            leads = leads[0]
        else:  # part of ST
            # ST7ssNJ0-PSG.edf where ss is the subject number, and N is the night.
            pid = int(ss - 82)
            if self.f:
                leads = 'ST7{:02d}{}J0-PSG_raw.fif'.format(pid, n)  # adjust for orig ST index (1 indexing)
            else:
                leads = 'ST7{:02d}{}J0-PSG.edf'.format(pid, n)  # adjust for orig ST index (1 indexing)
            rec_title = 'ST7{:02d}{}'.format(int(pid), n)
        if self.f:
            annotations = [x for x in os.listdir(db) if ((rec_title in x) and ('eve' in x))]
        else:
            annotations = [x for x in os.listdir(db) if ((rec_title in x) and ('Hypno' in x))]
        assert (len(annotations) == 1)
        annotations = annotations[0]
        leads = ''.join([db, leads])
        annotations = ''.join([db, annotations])
        try:
            assert (os.path.exists(leads))
        except AssertionError:
            print('leads path: ', leads)
            raise AssertionError
        assert (os.path.exists(annotations))
        return leads, annotations


class EEGTransform(object):
    """
    This will transform the EEG signal into a PyTorch tensor. FILTERING is
    expected to have OCCURED PREVIOUSLY or not required. Only a segment of
    the signal will be used as specified by the start and end times of the
    table. This is the place to apply other transformations as well, e.g.,
    normalization, etc.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, signal):
        """
        params:
              rosh: int
                  index of time series data where segment begins. if not provided use definition in function
              sof: int
                  index of time series data where segment ends. if not provided use definition in function
        """
        t_signal = signal.astype(np.single)
        # We transpose the signal to later use the lead dim as the channel... (C,L).
        t_signal = torch.reshape(torch.tensor(t_signal), (t_signal.size, 1))
        # ------^^^^^^^^^------#
        return t_signal  # Make sure I am a PyTorch Tensor


class EEGDataset(Dataset):

    def __init__(self, Transform, exp_table):
        super().__init__()  # When using a subclass, remember to inherit its properties.
        # ------Your code------#
        # Define self.exp_table (pandas from build experiment)
        # and self.transform (create an object from the transform we implemented):
        self.table = exp_table  # a panda
        self.transform = Transform()  # a function to get lead signal as proper torch
        self.minmax = pd.read_csv(os.getcwd() + "/mins_maxs.csv")  # panda of table with min and max of each lead
        try:
            # get seg_len
            seg_len = -1
            i = 0
            while seg_len < 0:  # seg_len must be pos result, could be negative if sample is around midnight
                start_arr = np.array([int(x) for x in self.table['srt'][i].split(':')])
                end_arr = np.array([int(x) for x in self.table['end'][i].split(':')])
                str_sec = (start_arr * np.array([3600, 60, 1])).sum()
                end_sec = (end_arr * np.array([3600, 60, 1])).sum()
                seg_len = int(end_sec - str_sec)
                i += 1
            self.seg_len = seg_len  # duration of segments in seconds
        except:
            self.seg_len = None

        # when called on index gives the file paths for signal and annotations
        self.get_rec_paths = PathFinder(filtered=False, tbl=self.table).get_paths
        self.asked_simple = 0  # did I already ask if this is simple table
        # ------^^^^^^^^^------#

    def get_label(self, index):
        # A method to decide the label:
        age = self.table['age'][index]
        sex = self.table['sex (F=1)'][index] - 1  # F1M2 switch to F0M1
        return sex, age  # F0M1

    def __getitem__(self, index):
        # ------Your code------#
        # Read the record with mne and transform its signal. Assign a label by using get_label.
        print(index)
        index = int(index)
        if index >= len(self):  # indices beyond range of samples return the first sample. This is only to fill out
            return self.__getitem__(0)  # the last batch in sequential loader/ evaluation it will not be used to train
        # Get data from specified recording
        rec_paths = self.get_rec_paths(index)  # path to lead file and annotation file

        # Get rosh and sof if in table
        meas_date = self.minmax['meas_date'][self.minmax['filename'] == rec_paths[0][-16:]].iloc[0]  # recpathshasindex
        try:
            rosh_arr = np.array([int(x) for x in self.table['srt'][index].split(':')])
            rosh_sec = (rosh_arr * np.array([3600, 60, 1])).sum()
            meas_sec = int(int(meas_date[-14:-12]) * 3600 + int(meas_date[-11:-9]) * 60 + int(meas_date[-8:-6]))
            if meas_sec > rosh_sec:
                rosh = (rosh_sec + (24 * 3600 - meas_sec))
            else:
                rosh = rosh_sec - meas_sec
            sof = rosh + self.seg_len - .01   # seg_len is inferred from tabular data minus .01 since using time not ind

        except (NameError, KeyError):  # error is rosh is not defined
            if not self.asked_simple:
                self.asked_simple += 1
                print('is this simple path ?')  # should only see if there are no segment times specified
            rosh = 3600
            sof = 3600 + 30 - .01
            pass

        # print('rosh (sec): ', rosh)
        # print('sof (sec): ', sof)

        with contextlib.redirect_stdout(io.StringIO()) as f:
            raw = mne.io.read_raw_edf(rec_paths[0])
            raw.crop(rosh, sof)
            ch = ['EEG Fpz-Cz', 'EEG Pz-Oz']
            raw.pick_channels(ch, ordered=True)
            raw.load_data()
            raw = raw.filter(l_freq=1, h_freq=49)
            data1 = raw[ch[0]][0]  # np.ndarray (1,signal length)
            data2 = raw[ch[1]][0]
            raw.close()
        # min0 = self.minmax['min0'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        # min1 = self.minmax['min1'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        # max0 = self.minmax['max0'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        # max1 = self.minmax['max1'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        # data = np.array([(data1 - min0) / (max0 - min0), (data2 - min1) / (max1 - min1)])
        mean0 = self.minmax['mean0'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        mean1 = self.minmax['mean1'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        std0 = self.minmax['std0'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        std1 = self.minmax['std1'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
        data = np.array([(data1 - mean0) / std0, (data2 - mean1) / std1])

        # Apply the transform to get the right shape and data type
        fpzczt = self.transform(data[0, :])
        signal = torch.zeros(2, len(fpzczt))  # use transformed lead length to make signal array #zero divison concern?
        signal[0, :] = fpzczt.squeeze()
        signal[1, :] = (self.transform(data[1, :])).squeeze()  # get and transform same line to save memory

        label = self.get_label(index)
        #removed in rewrite# signal = torch.tensor(signal)
        signal = signal.to(dtype=torch.double)
        # ------^^^^^^^^^------#
        return signal, label

    def __len__(self):
        return len(self.table)  # database size


# not to be used # def filter_save(overwrite):

def segment_hours(tbl_fn, seg_len=30, stages={'Sleep stage 1': 2}, limit=False):
    # seg_len: segment length in seconds
    # stages is dictionary of sleep stages to use
    # tbl_fn: file name of the table
    # this function creates a csv file which contains the start and end hours of each segment
    s_tbl_path = 'simple table.csv'
    tbl = pd.read_csv(s_tbl_path, header=0, index_col=0)
    new_tbl = pd.DataFrame(columns=['subject', 'night', 'age', 'sex (F=1)', 'LightsOff'])
    getPaths = PathFinder(filtered=False, tbl=build_simple_table()).get_paths
    for rec in range(197):
        paths = getPaths(rec)
        with contextlib.redirect_stdout(io.StringIO()) as f:
            raw0 = mne.io.read_raw_edf(paths[0], stim_channel='Event marker', misc=['Temp rectal'])
            annot0 = mne.read_annotations(paths[1])  # original annotations
            raw0.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])  # irrelevant (?)
            raw0.set_annotations(annot0, emit_warning=False)
        # Find areas of high eog activity
        #####################################
        # remove annotations where there is determined to be much eog activity
        # removes annotations from outside given time range use copy so as not to edit in place
        # annot_train.crop(annot_train[1]['onset'] - 60 * 3, annot_train[-2]['onset'] + 30)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                events_train, _ = mne.events_from_annotations(raw0, event_id=stages, chunk_duration=seg_len)
            print('events_train shape : ', events_train.shape)
        except ValueError: # when there are no events
            print('recording: ', rec, ' has no segments matching requested')
            continue
        # if a limit was chosen shuffle the found segments and choose only that many

        if (events_train.shape[0] > limit) and limit:  # use all segments if no. segments found is less than the limit
            np.random.shuffle(events_train)  # acts in place
            events_train = events_train[0:limit, :]
        event_onsets = events_train[:, 0] / 100   # np.ndarray # from time sample to sec since start of rec
        # should be equal to limit now
        # print('3432', event_onsets.shape[0])
        if event_onsets.shape[0] != limit:
            print(paths[0])
            print('event_onsets shape: ', event_onsets.shape)
            # print('events_train: ', events_train)
        meas_date = raw0.info['meas_date']
        time_in_seconds = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second + event_onsets  # the seconds passed since 00:00:00 midnight
        time_in_seconds[time_in_seconds >= 24 * 3600] = time_in_seconds[time_in_seconds >= 24 * 3600] - (24 * 3600)

        time_in_seconds_end = time_in_seconds + seg_len
        time_in_seconds_end[time_in_seconds_end >= 24 * 3600] = time_in_seconds_end[time_in_seconds_end >= 24 * 3600] - (24 * 3600)

        epoch_start = np.array([str(datetime.timedelta(seconds=s)) for s in time_in_seconds])  # an array of many HH:MM:SS
        epoch_end = np.array([str(datetime.timedelta(seconds=s)) for s in time_in_seconds_end])
        epoch_times = np.vstack((epoch_start, epoch_end)).T
        epoch_times = pd.DataFrame(epoch_times, columns=['srt', 'end'])
        # ---------------
        n_segs = epoch_times.shape[0]
        timespd = pd.DataFrame(epoch_times, columns=['srt', 'end'])
        rec_rows = tbl.loc[[rec]*n_segs]
        rec_rows = rec_rows.reset_index(drop=True)
        rec_rows = rec_rows.join(timespd)
        new_tbl = pd.concat([new_tbl, rec_rows]).reset_index(drop=True)
        if (rec % 20) == 0:
            print(rec/len(tbl), '%')
    new_tbl.to_csv(tbl_fn)
    return new_tbl


def build_experiment_tbl(experiment_fn):
    # if output is saved does not rerun
    try:
        extable = pd.read_csv(experiment_fn, header=0, index_col=0)
        print("{} Already exists".format(experiment_fn))
        return extable
    except:
        print("{} did not already exist".format(experiment_fn))
        print("creating now")
    # verify simple table is built
    build_simple_table()
    experiments = {
        "sleep_1_05sec_lim1.csv": {'tbl_fn': "sleep_1_05sec_lim1.csv", 'seg_len': 0.5, 'stages': {'Sleep stage 1': 2},
                                   'limit': 1},
        "sleep_1_05sec_lim180.csv": {'tbl_fn': "sleep_1_05sec_lim180.csv", 'seg_len': 0.5,
                                     'stages': {'Sleep stage 1': 2}, 'limit': 180},
        "sleep_1_05sec_lim720.csv": {'tbl_fn': "sleep_1_05sec_lim720.csv", 'seg_len': 0.5,
                                     'stages': {'Sleep stage 1': 2}, 'limit': 720},
        "sleep_1_2sec_lim10.csv": {'tbl_fn': "sleep_1_2sec_lim10.csv", 'seg_len': 2, 'stages': {'Sleep stage 1': 2},
                                   'limit': 10},
        "sleep_1_2sec_lim75.csv": {'tbl_fn': "sleep_1_2sec_lim75.csv", 'seg_len': 2, 'stages': {'Sleep stage 1': 2},
                                   'limit': 75},
        "sleep_1_5sec_lim10.csv": {'tbl_fn': "sleep_1_5sec_lim10.csv", 'seg_len': 5, 'stages': {'Sleep stage 1': 2},
                                   'limit': 10},
        "sleep_1_5sec_lim20.csv": {'tbl_fn': "sleep_1_5sec_lim20.csv", 'seg_len': 5, 'stages': {'Sleep stage 1': 2},
                                   'limit': 20},
        "sleep_1_5sec_lim30.csv": {'tbl_fn': "sleep_1_5sec_lim30.csv", 'seg_len': 5, 'stages': {'Sleep stage 1': 2},
                                   'limit': 30},
        "sleep_1_5sec_lim50.csv": {'tbl_fn': "sleep_1_5sec_lim50.csv", 'seg_len': 5, 'stages': {'Sleep stage 1': 2},
                                   'limit': 50},
        "sleep_1_10sec.csv": {'tbl_fn': "sleep_1_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage 1': 2}},
        "sleep_1_10sec_lim15.csv": {'tbl_fn': "sleep_1_10sec_lim15.csv", 'seg_len': 10, 'stages': {'Sleep stage 1': 2},
                                    'limit': 15},
        "sleep_1_30sec.csv": {'tbl_fn': "sleep_1_30sec.csv", 'seg_len': 30, 'stages': {'Sleep stage 1': 2}},
        "sleep_1_30sec_lim5.csv": {'tbl_fn': "sleep_1_30sec_lim5.csv", 'seg_len': 30, 'stages': {'Sleep stage 1': 2},
                                   'limit': 5},
        "sleep_1_30sec_lim180.csv": {'tbl_fn': "sleep_1_30sec_lim180.csv", 'seg_len': 30,
                                     'stages': {'Sleep stage 1': 2}, 'limit': 180},
        "sleep_1_45sec.csv": {'tbl_fn': "sleep_1_45sec.csv", 'seg_len': 45, 'stages': {'Sleep stage 1': 2}},
        "sleep_1_60sec.csv": {'tbl_fn': "sleep_1_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage 1': 2}},
        "sleep_1_120sec.csv": {'tbl_fn': "sleep_1_120sec.csv", 'seg_len': 120, 'stages': {'Sleep stage 1': 2}},
        "sleep_1_240sec.csv": {'tbl_fn': "sleep_1_2400sec.csv", 'seg_len': 240, 'stages': {'Sleep stage 1': 2}},
        "sleep_1_480sec.csv": {'tbl_fn': "sleep_1_4800sec.csv", 'seg_len': 480, 'stages': {'Sleep stage 1': 2}},
        "sleep_2_05sec_lim1.csv": {'tbl_fn': "sleep_2_05sec_lim1.csv", 'seg_len': 0.5, 'stages': {'Sleep stage 2': 2},
                                   'limit': 1},
        "sleep_2_05sec_lim3.csv": {'tbl_fn': "sleep_2_05sec_lim3.csv", 'seg_len': 0.5, 'stages': {'Sleep stage 2': 2},
                                   'limit': 3},
        "sleep_2_05sec_lim180.csv": {'tbl_fn': "sleep_2_05sec_lim180.csv", 'seg_len': 0.5,
                                     'stages': {'Sleep stage 2': 2}, 'limit': 180},
        "sleep_2_05sec_lim720.csv": {'tbl_fn': "sleep_2_05sec_lim720.csv", 'seg_len': 0.5,
                                     'stages': {'Sleep stage 2': 2}, 'limit': 720},
        "sleep_2_10sec.csv": {'tbl_fn': "sleep_2_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage 2': 2}},
        "sleep_2_10sec_lim385.csv": {'tbl_fn': "sleep_2_10sec_lim385.csv", 'seg_len': 10,
                                     'stages': {'Sleep stage 2': 2}, 'limit': 385},
        "sleep_2_30sec.csv": {'tbl_fn': "sleep_2_30sec.csv", 'seg_len': 30, 'stages': {'Sleep stage 2': 2}},
        "sleep_2_30sec_lim180.csv": {'tbl_fn': "sleep_2_30sec_lim180.csv", 'seg_len': 30,
                                     'stages': {'Sleep stage 2': 2}, 'limit': 180},
        "sleep_2_30sec_lim90.csv": {'tbl_fn': "sleep_2_30sec_lim90.csv", 'seg_len': 30, 'stages': {'Sleep stage 2': 2},
                                    'limit': 90},
        "sleep_2_45sec.csv": {'tbl_fn': "sleep_2_45sec.csv", 'seg_len': 45, 'stages': {'Sleep stage 2': 2}},
        "sleep_2_60sec.csv": {'tbl_fn': "sleep_2_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage 2': 2}},
        "sleep_2_120sec.csv": {'tbl_fn': "sleep_2_120sec.csv", 'seg_len': 120, 'stages': {'Sleep stage 2': 2}},
        "sleep_2_240sec.csv": {'tbl_fn': "sleep_2_240sec.csv", 'seg_len': 240, 'stages': {'Sleep stage 2': 2}},
        "sleep_2_480sec.csv": {'tbl_fn': "sleep_2_480sec.csv", 'seg_len': 480, 'stages': {'Sleep stage 2': 2}},
        "sleep_3_5sec.csv": {'tbl_fn': "sleep_3_5sec.csv", 'seg_len': 5, 'stages': {'Sleep stage 3': 2}},
        "sleep_3_10sec.csv": {'tbl_fn': "sleep_3_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage 3': 2}},
        "sleep_4_10sec.csv": {'tbl_fn': "sleep_4_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage 4': 2}},
        "sleep_R_10sec.csv": {'tbl_fn': "sleep_R_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage R': 2}},
        "sleep_R_30sec.csv": {'tbl_fn': "sleep_R_30sec.csv", 'seg_len': 30, 'stages': {'Sleep stage R': 2}},
        "sleep_R_45sec.csv": {'tbl_fn': "sleep_R_45sec.csv", 'seg_len': 45, 'stages': {'Sleep stage R': 2}},
        "sleep_R_60sec.csv": {'tbl_fn': "sleep_R_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage R': 2}},
        "sleep_R_120sec.csv": {'tbl_fn': "sleep_R_120sec.csv", 'seg_len': 120, 'stages': {'Sleep stage R': 2}},
        "sleep_R_240sec.csv": {'tbl_fn': "sleep_R_240sec.csv", 'seg_len': 240, 'stages': {'Sleep stage R': 2}},
        "sleep_R_480sec.csv": {'tbl_fn': "sleep_R_480sec.csv", 'seg_len': 480, 'stages': {'Sleep stage R': 2}},
        "sleep_W_10sec.csv": {'tbl_fn': "sleep_W_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage W': 2}},
        "sleep_W_30sec.csv": {'tbl_fn': "sleep_W_30sec.csv", 'seg_len': 30, 'stages': {'Sleep stage W': 2}},
        "sleep_W_45sec.csv": {'tbl_fn': "sleep_W_45sec.csv", 'seg_len': 45, 'stages': {'Sleep stage W': 2}},
        "sleep_W_60sec.csv": {'tbl_fn': "sleep_W_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage W': 2}},
        "sleep_W_120sec.csv": {'tbl_fn': "sleep_W_120sec.csv", 'seg_len': 120, 'stages': {'Sleep stage W': 2}},
        "sleep_W_240sec.csv": {'tbl_fn': "sleep_W_240sec.csv", 'seg_len': 240, 'stages': {'Sleep stage W': 2}},
        "sleep_W_480sec.csv": {'tbl_fn': "sleep_W_480sec.csv", 'seg_len': 480, 'stages': {'Sleep stage W': 2}},
        "sleep_123_60sec.csv": {'tbl_fn': "sleep_123_60sec.csv", 'seg_len': 60,
                                'stages': {'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3}}}

    assert(experiment_fn == experiments[experiment_fn]['tbl_fn'])
    new_tbl = segment_hours(**experiments[experiment_fn])
    return new_tbl


def train_val_test_split_by_pid(dataset, batch_size, train_rt=.6, val_rt=.2,  num_workers=0, verbose=1):
    """
    Finds a split within 2 percent of split specified that keeps patients in test set out of train set.
    :param dataset: PyTorch dataset to split.
    :param train_rt: Split ratio for the data - e.g., 0.8 ==> 80% train and 20% spilt between test and validation
    :param val_rt: Split ratio for the data - e.g., 0.1 ==> 10% validation
    :param batch_size: Define the batch_size to use in the DataLoaders.
    :param num_workers: Define the num_workers to use in the DataLoaders.
    :return: train and test DataLoaders.
    """
    # ------Your code------#
    # since some patients have less data must verify that this split is adequate
    acptE = 0.005  # 1/2 percent
    error_tr = error_vl = 1
    iteration = 0
    pid_samp = dataset.table['subject']  # patient id for each sample as in table
    u_pids = np.unique(pid_samp)  # list of unique patient ids
    npt = len(u_pids)
    while np.logical_or(error_tr > acptE, error_vl > acptE):
        # change number on different iterations for more randomness and higher chance of finding a good split
        n_test_pt = int(round(npt*(1-train_rt-val_rt))+np.random.randint(-3, 3, size=(1,), dtype=int))
        # ensures that pts with only one recording are not always in the test set
        n_val_pt = int(round(npt*val_rt)+np.random.randint(-3, 3, size=(1,), dtype=int))
        random_order = np.random.permutation(u_pids)
        train_pts = random_order[:-(n_test_pt+n_val_pt)]
        test_pts = random_order[-(n_test_pt+n_val_pt):-n_val_pt]
        val_pts = random_order[-n_val_pt:]
        # use test pts to get indexes of samples for those patients, shuffle them, and pass them to a data loader
        # should only be one d # need the first index because it was returning a tuple now ndarray(number,)
        test_samp_indexes = np.where(np.isin(pid_samp, test_pts))[0]
        dl_test = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                             sampler=SubsetRandomSampler(np.random.permutation(test_samp_indexes)))
        # validation
        # should only be one d # need the first index because it was returning a tuple now ndarray(number,)
        val_samp_indexes = np.where(np.isin(pid_samp, val_pts))[0]
        dl_val = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                            sampler=SubsetRandomSampler(np.random.permutation(val_samp_indexes)))
        # train
        train_samp_indexes = np.where(np.isin(pid_samp, train_pts))[0]  # should only be one d
        dl_train = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                              sampler=SubsetRandomSampler(np.random.permutation(train_samp_indexes)))
        # calculate the split ratios for evaluation
        actual_tr_split = len(dl_train.sampler.indices)/dataset.__len__()
        actual_vl_split = len(dl_val.sampler.indices)/dataset.__len__()
        error_tr = abs(train_rt - actual_tr_split)
        error_vl = abs(val_rt - actual_vl_split)
        # since some patients have less data must verify that this split is adequate
        iteration += 1
        if iteration == 100:
            print("could not find a good split after 100 tries")
            print("error tr = {:.4f}".format(error_tr))
            print("error vl = {:.4f}".format(error_vl))
            break
        if verbose > 1:
            print('iteration', iteration)
            print('dl_train', len(dl_train.sampler.indices))
            print('dl_test', len(dl_test.sampler.indices))
            print('dl_val', len(dl_val.sampler.indices))
            print('actual tr split {:.4f}'.format(actual_tr_split))
            print('actual vl split {:.4f}'.format(actual_vl_split))
            print('actual tst split {:.4f}'.format(len(dl_test.sampler.indices)/dataset.__len__()))
    if verbose:
        print('n_samps dl_train', len(dl_train.sampler.indices))
        print('n_samps dl_test', len(dl_test.sampler.indices))
        print('n_samps dl_val', len(dl_val.sampler.indices))
        print('actual tr split {:.4f}'.format(actual_tr_split))
        print('actual tst split {:.4f}'.format(len(dl_test.sampler.indices) / dataset.__len__()))
        print('actual vl split {:.4f}'.format(actual_vl_split))
    # ------^^^^^^^^^------#
    return dl_train, dl_test, dl_val


# Training loop
def forward_epoch(model, dl, loss_function, optimizer, total_loss=0,
                  to_train=False, desc=None, device=torch.device('cpu'), label=0):

    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        model = model.double().to(device)  # solving runtime memory issue

        y_trues = torch.empty(0).type(torch.int).to(device)
        y_preds = torch.empty(0).type(torch.int).to(device)
        for i_batch, (X, y) in enumerate(dl):
            X = X.to(device)  # TODO the line that David changed and saw decent results
            # try blstm run with 10000 loss .769 acc .489 note all classed males sex val .462
            # try blstm run with 5000 loss acc note
            X = X.type(torch.double)

            if str(type(dl.dataset)) == "<class 'aux_eeg_rewrite.EEGDataset_variable_creation'>":
                y = y[:, label].to(device)  # added index because of get label returning sex, age
            else:
                y = y[label].to(device)  # added index because of get label returning sex, age
            # Forward:
            y_pred = model(X)

            # Loss:
            y_true = y.type(torch.double)  # changed for ce_w
            # print('ytrue shape:', y_true.shape)
            # print('ypred shape:', y_pred.shape) # gettting a RunTime error on last batch because y_pred and ytrue different sizes
            loss = loss_function(y_pred, y_true)  # mean loss of one batch
            total_loss += loss.item()  # must be one ele to add item. It is becuase we use reduction mean of batch
            if i_batch == len(dl)-1:  # solution to zero dimensional issue in torch.cat
                if y_true.size() == torch.Size([]):
                    y_true = np.expand_dims(y_true, 0)
                    y_true = torch.tensor(y_true)
                if y_pred.size() == torch.Size([]):
                    y_pred = np.expand_dims(y_pred, 0)
                    y_pred = torch.tensor(y_pred)

            y_trues = torch.cat((y_trues, y_true))
            y_preds = torch.cat((y_preds, y_pred))

            if to_train:
                # Backward:
                loss = loss_function(y_pred, y_true)  # mean loss of one batch # Rewrite
                optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
                loss.backward()  # get gradient of loss

                # Optimization step:
                optimizer.step()  # use gradients to update weights

            # Progress bar:
            pbar.update(1)

    return total_loss, y_trues, y_preds


def name_dir(run_desc):
    # only will work on server
    # get last named dir
    if not 'stu16' in os.getcwd():
        dir_path = None
    else:
        dir_name_file = open("/home/stu16/EEG_proj/runs/last_dir", "r")
        last_num = int(dir_name_file.read())
        dir_name_file.close()
        # get next dir name
        dir_num = last_num + 1
        # creat new dir for the current run
        dir_path = "/home/stu16/EEG_proj/runs/{:03d}/".format(dir_num)
        os.mkdir(dir_path)
        # open text file-store the last directory created
        dir_name_file = open("/home/stu16/EEG_proj/runs/last_dir", "w")
        # write string to file
        n = dir_name_file.write('{:03d}'.format(dir_num))
        dir_name_file.close()
        # open text file-store directory description
        dir_desc_file = open(dir_path+"/run_desc", "w")
        # write description string to file
        n = dir_desc_file.write(run_desc)
        dir_desc_file.close()
    return dir_path


def append_desc(desc, dir_path):
    with open(dir_path+"run_desc", "a") as f:
        f.write(desc)


class my_weighted_bce():
    def __init__(self, weight: torch.tensor):
        self.w = weight

    def __call__(self, y_pred: torch.tensor, y_true: torch.tensor):
        # weight proportion of nMajority/nMin should be for dist in training only
        w_bce = torch.mean(-(self.w*(y_true*torch.log(y_pred))+(1-y_true)*torch.log(1-y_pred)))
        return w_bce


class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Define self.direct_path by adding the layers into a nn.Sequential. Use nn.Conv1d and nn.Relu.
        # You can use padding to avoid reducing L size, to allow the skip-connection adding.
        # ------Your code------#
        self.direct_path = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        )
        # ------^^^^^^^^^------#

        # Define self.skip_layers path.
        # You should use convolution layer with a kernel size of 1 to consider the case where the input and output shapes mismatch.
        # ------Your code------#
        skip_layers = []
        if in_channels != 64:  # HOW DOES THIS PART WORK? When are you adding the layers
            skip_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1,
                          bias=False)
            )
        else:
            self.skip_path = nn.Sequential(*skip_layers)
        # ------^^^^^^^^^------#

    def forward(self, x):
        # ------Your code------#
        # Compute the two paths and add the results to each other, then use ReLU (torch.relu) to activate the output.
        direct_output = self.direct_path(x)
        skip_output = self.skip_path(x)
        activated_output = torch.relu(direct_output + skip_output)
        # ------^^^^^^^^^------#
        return activated_output


class LSTM_adapted(nn.LSTM):
    # only return the out
    def __init__(self, input_size=500, hidden_size=256, batch_first=True, num_layers=2, dropout=0.2,
                 bidirectional=True):
        super().__init__(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_layers, dropout=dropout,
                 bidirectional=bidirectional)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
    def forward(self, x):
        out, states = self.lstm(x.double())
        return out


def dataviz(batch_sz, dl_train, show=False):
    # show data and returns last batch to show as signals
    batch_to_show = 2
    plt.figure(figsize=(18, 6))
    for i, (signals, labels) in enumerate(dl_train):  # signals.shape torch.Size([4, 2, 60000])
        # break  # skip dat viz for now
        if not show:
            return signals
        if str(type(dl_train.dataset))=="<class 'aux_eeg_rewrite.EEGDataset_variable_creation'>":
            labels = labels[:, 0]
        else:
            labels = labels[0]
        for j in range(batch_sz):
            plt.subplot(batch_to_show, batch_sz, j + 1 + (i * batch_sz))
            plt.imshow(signals[j][:, :200])
            plt.title(labels[j].item())
            plt.axis('off')
        if i + 1 == batch_to_show:
            break
    if show:
        plt.show()
    return signals


def sex_split_bar(eeg_ds, dls, groups, dir_path, show=False):
    for dl_i, dl in enumerate(dls):
        # Plot Sex bar plots
        plt.figure(figsize=(18, 6))
        nF = sum(eeg_ds.table['sex (F=1)'][dl.sampler.indices] == 1)  # by segment
        nM = sum(eeg_ds.table['sex (F=1)'][dl.sampler.indices] == 2)
        bars = plt.bar(x=(0, 1), height=(nF, nM))   # in table F1M2 in label F0M1
        plt.bar_label(bars, labels=('F ({:.2f}%)'.format(100*nF/(nF+nM)), 'M ({:.2f}%)'.format(100*nM/(nF+nM))))
        plt.xlabel('sex'), plt.ylabel('count by segment'), plt.title(groups[dl_i])
        if dir_path:
            plt.savefig(dir_path + "sexBar_" + groups[dl_i] + ".png")
        if show:
            plt.show()  # Plot to show the representation of each pt in group; bar plot


def age_split_hist(eeg_ds, dls, groups, dir_path, show=False):
    # Plot age hists
    for dl_i, dl in enumerate(dls):
        plt.figure() #(figsize=(18, 6))
        plt.hist(eeg_ds.table['age'][dl.sampler.indices])  # by segment
        plt.xlabel('age'), plt.ylabel('count by segment'), plt.title(groups[dl_i])
        if dir_path:
            plt.savefig(dir_path+"ageHist_" + groups[dl_i]+".png")
        if show:
            plt.show()


def pt_split_tbl(eeg_ds, dls, groups, dir_path):
    # Table of patients by split group.
    # Get first occurence of patient in table
    recs = (eeg_ds.table['subject']*2 + eeg_ds.table['night']).array  # col where all recs have unique value
    pts_by_group = pd.DataFrame()
    for dl_i, dl in enumerate(dls):
        if dl_i == 0:
            pts_by_group = pd.DataFrame()
        pts_dl = np.unique(eeg_ds.table['subject'][dl.sampler.indices])  # patients in group
        pts_df = pd.DataFrame(pts_dl)
        pts_df.columns = [groups[dl_i]]
        pts_by_group = pd.concat((pts_by_group, pts_df), axis=1)  # Table of which patients in which group
        # first_i_pt = [ind for ind in dl.sampler.indices
        #               if ind == np.where(eeg_ds.table['subject'] == eeg_ds.table['subject'][ind])[0][0]]
        # first_i_pt.sort()  # the indexes of the first occurrences of each unique pt in the dl
        # first_i_rec = [ind for ind in dl.sampler.indices if ind == np.where(recs == recs[ind])[0][0]]
        # first_i_rec.sort()  # the indexes of the first occurrences of each unique record in the dl
        # # save panda of patient by group split
        if dir_path:
            pts_by_group.to_csv(dir_path + "ptSplit.csv")
        return pts_by_group


def sort_pred_by_rec(eeg_ds, y_pred, y_true, s1a0):
    # sort the predictions from dl_all by the rec number in simple table
    # assumes the predictions share indexing scheme of the experiment table
    # mean works for sex and age but mode only makes sense for sex
    # s1a0 is flag 1 for sex 0 for age
    with contextlib.redirect_stdout(io.StringIO()) as f:
        st = build_simple_table()
    pred_c = (y_pred > 0.5) * 1  # classified prediction as one or zero
    rec_preds_p = [None] * 197  # for all pt ids 0-106 and 2 nights by probability
    rec_preds_c = [None] * 197  # for all pt ids 0-106 and 2 nights by class
    for exp_ind in range(len(eeg_ds.table)):
        sub = eeg_ds.table['subject'][exp_ind]
        night = eeg_ds.table['night'][exp_ind]
        l = st.index[(st['subject'] == sub) & (st['night'] == night)].tolist()
        assert len(l) == 1
        rec_num = l[0]
        if s1a0:  # confirm true labels match
            assert y_true[exp_ind] == (st['sex (F=1)'][rec_num]-1)
            rec_true = np.array([truth - 1 for truth in st['sex (F=1)']])
        else:
            assert y_true[exp_ind] == st['age'][rec_num]
            rec_true = np.array([truth for truth in st['age']])
        rec_preds_p[rec_num] = np.array([y_pred[exp_ind]]) if rec_preds_p[rec_num] is None else np.concatenate(
            (rec_preds_p[rec_num], np.array([y_pred[exp_ind]])))
        rec_preds_c[rec_num] = np.array([pred_c[exp_ind]]) if rec_preds_c[rec_num] is None else np.concatenate(
            (rec_preds_c[rec_num], np.array([pred_c[exp_ind]])))
    return rec_preds_p, rec_preds_c, rec_true


def sort_pred_by_dl(eeg_ds, dls, rec_preds_p, rec_preds_c, s1a0=1):
    # sort the predictions by the dl they are from. dls train, test, val
    et = eeg_ds.table  # experiment table
    with contextlib.redirect_stdout(io.StringIO()) as f:
        st = build_simple_table()
    for it, dl in enumerate(dls):
        dl_is = dl.sampler.indices
        recs = [st.index[(st['subject'] == et['subject'][ind]) & (st['night'] == et['night'][ind])].tolist()[0] for ind in dl_is]
        recs = np.unique(recs) # was list now ndarray
        means = [np.mean(rec_preds_p[rec]) for rec in recs]
        modes = [0 if np.count_nonzero(rec_preds_c[rec] == 0) >
                     np.count_nonzero(rec_preds_c[rec] == 1) else 1 for rec in recs]  # give tie to 1 (men)
        truths = [st['sex (F=1)'][rec]-1 for rec in recs]
        if not s1a0: # age
            truths = [st['age'][rec] for rec in recs]
        if it == 0:
            recs_dltr, means_tr, modes_tr, true_tr = recs, means, modes, truths
        elif it == 1:
            recs_dltst, means_tst, modes_tst, true_tst = recs, means, modes, truths
        elif it == 2:
            recs_dlval, means_val, modes_val, true_val = recs, means, modes, truths
    return (recs_dltr, means_tr, modes_tr, true_tr), \
           (recs_dltst, means_tst, modes_tst, true_tst), \
           (recs_dlval, means_val, modes_val, true_val)


class EEGDataset_variable_creation(Dataset):

    def __init__(self, Transform, exp_table, table_name, filtered=0):  # TODO changed to see effect on loading
        super().__init__()  # Subclass so inherit properties.
        self.transform = Transform()  # instance of Transform object we implemented
        self.minmax = pd.read_csv(os.getcwd() + "/mins_maxs.csv")  # panda of table with min/max/mean/std of each lead
        self.table = exp_table        # pandas from build experiment
        self.tableName = table_name.replace('.csv', '')   # str of table name
        self.num_sig_f = 0   # counts the number of signals variables that were created
        self.iterator2 = 0  # counts the number of calls to __getitem__
        self.signals = 0
        self.labels = 0
        self.indices = 0
        try:
            # get seg_len
            seg_len = -1
            i = 0
            while seg_len < 0:
                start_arr = np.array([int(x) for x in self.table['srt'][i].split(':')])
                end_arr = np.array([int(x) for x in self.table['end'][i].split(':')])
                str_sec = (start_arr * np.array([3600, 60, 1])).sum()
                end_sec = (end_arr * np.array([3600, 60, 1])).sum()
                seg_len = int(end_sec - str_sec)
                i += 1
            self.seg_len = seg_len  # duration of segments in seconds
        except:
            self.seg_len = None
        # self.get_rec_paths is callable on index gives the file paths for signal and annotations
        self.get_rec_paths = PathFinder(filtered=filtered, tbl=self.table).get_paths
        self.asked_simple = 0  # did I already ask if this is simple table

        if os.path.exists(f'{self.tableName}_all_signals.npy'):  # if var exists then use pass otherwise create it
            self.variable_made = True
        else:  # create the variable
            self.variable_made = False
            self.create_var()
            self.variable_made = True
        if self.variable_made:
            with open(f'{self.tableName}_all_labels.npy', 'rb') as f:
                self.labels = np.load(f, allow_pickle=True)
            with open(f'{self.tableName}_all_indices.npy', 'rb') as f:
                self.indices = np.load(f, allow_pickle=True)
            with open(f'{self.tableName}_all_signals.npy', 'rb') as f:
                self.signals = np.load(f, allow_pickle=True)

    def get_label(self, index):
        # A method to decide the label:
        age = self.table['age'][index]  # TODO return age label as well
        sex = self.table['sex (F=1)'][index] - 1  # F1M2 switch to F0M1
        return sex, age  # F0M1

    def __getitem__(self, index):
        # Read the record with mne. Filter and transform its signal. Assign a label by using get_label.
        index = int(index)
        if index >= len(self):  # indices beyond range of samples return the first sample. This is only to fill out
            return self.__getitem__(0)  # the last batch in sequential loader/ evaluation it will not be used to train

        if self.variable_made:
            newIndex = np.where(self.indices == index)[0][0]  # find the place of the index
            signal = self.signals[:, newIndex, :]
            label = self.labels[newIndex, :]
            return signal, label  # returning numpy arrays which are converted to tensors in the training loop

        elif not self.variable_made:  # file containing all of the samples doesn't exist, create it
            # Read the record with mne. Filter and transform its signal. Assign a label by using get_label.
            index = int(index)
            if index % 999 == 0:
                print('processed ', index+1, ' records: ', index/len(self))
            # Get data from specified recording
            rec_paths = self.get_rec_paths(index)
            # start = time()
            with contextlib.redirect_stdout(io.StringIO()) as f:
                raw = mne.io.read_raw_edf(rec_paths[0])
                meas_date = raw.info['meas_date']
            try:
                rosh_arr = np.array([float(x) for x in self.table['srt'][index].split(':')])  # arr hour, min, sec
                rosh_sec = (rosh_arr * np.array([3600, 60, 1])).sum()  # secs since midnight
                meas_sec = (meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second)  # secs since midnight
                if meas_sec > rosh_sec:  # date correction if meas_date is later than start of segment
                    rosh = (rosh_sec + (24 * 3600 - meas_sec))  # ie meas before midnight srt after midnight
                else:    # no date correction needed         rosh: time in secs from recording to where seg starts
                    rosh = (rosh_sec - meas_sec)
                # seg_len is inferred from tabular data minus .01 bc by time (100hz) not index
                sof = rosh + self.seg_len - .01  # time in secs from recording to where seg ends
            except (NameError, KeyError):  # error is rosh is not defined
                if not self.asked_simple:
                    self.asked_simple += 1
                    print('is this simple path ?')  # should only see if there are no segment times specified
                rosh = 3600
                sof = 3600 + 30 - .01  # default to 30 sec segment

            raw.crop(rosh, sof)  # crop by rosh and sof now to save memory. raw.crop uses seconds not index
            # TODO filter/normalize here
            ch = ['EEG Fpz-Cz', 'EEG Pz-Oz']
            raw.pick_channels(ch, ordered=True)
            with contextlib.redirect_stdout(io.StringIO()) as f:
                raw.load_data()
                raw = raw.filter(l_freq=1, h_freq=49)
            data0 = raw[ch[0]][0]  # np.ndarray (1,signal length)
            data1 = raw[ch[1]][0]
            raw.close()
            mean0 = self.minmax['mean0'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
            mean1 = self.minmax['mean1'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
            std0 = self.minmax['std0'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
            std1 = self.minmax['std1'][self.minmax['filename'] == rec_paths[0][-16:]].to_numpy()
            data0 = np.array((data0 - mean0) / std0)
            data1 = np.array((data1 - mean1) / std1)
            # TODO filtering
            # Apply the transform to get the right shape and data type
            # channels are tuples with ind0 containing the signal and ind1 holding the times in seconds
            # channel names ['EEG Fpz-Cz', 'EEG Pz-Oz']
            fpzczt = self.transform(data0)
            del data0
            signal = np.zeros((2, len(fpzczt)))  # use transformed lead len to make signal arr
            signal[0, :] = fpzczt.squeeze()      # get and transform same line to save memory
            del fpzczt
            signal[1, :] = (self.transform(data1)).squeeze()
            del data1
            signal = torch.tensor(signal)
            signal = signal.to(dtype=torch.double)
            label = self.get_label(index)
            return signal, label

    def create_var(self):  # TODO added for the file creation:
        for index in range(self.__len__()):
            signal, label = self.__getitem__(index)
            # if it's the first call of __getitem__ then don't concatenate. Rather, assign a value instead
            if index == 0 or (index % 999 == 1 and index != 1):
                self.signals = signal  # signals is large so needs to be chunked labels/indices don't
            if index == 0:
                self.labels = np.array([label])
                self.indices = np.array([index])
            else:  # index != 0:if it's not the first call, concatenate
                self.labels = np.concatenate((self.labels, np.array([label])))
                self.indices = np.concatenate((self.indices, np.array([index])))
                try:  # try concat if issue save prev collection of sigs. current sig is concated with next one
                    self.signals = torch.cat((self.signals, signal))
                    if index % 999 == 0:  # every 1000 iterations save the variables and delete them so we don't concatenate huge variables
                        np.save(f'{self.tableName}_{self.num_sig_f}_indices.npy', self.indices)
                        np.save(f'{self.tableName}_{self.num_sig_f}_signals.npy', self.signals)
                        np.save(f'{self.tableName}_{self.num_sig_f}_labels.npy', self.labels)
                        if self.num_sig_f != 0:
                            os.remove(
                                f'{self.tableName}_{self.num_sig_f - 1}_indices.npy')  # deleting the shorter ver of file
                            os.remove(f'{self.tableName}_{self.num_sig_f - 1}_labels.npy')
                        del self.signals
                        self.num_sig_f += 1
                except:  # if it has unsolvable mem issues it saves the curr var, deletes it and assigns a new val
                    np.save(f'{self.tableName}_{self.num_sig_f}_signals.npy', self.signals)
                    del self.signals
                    print('Not enough memory. Saved tensor.')
                    self.signals = signal
                    self.num_sig_f += 1

            if index == self.__len__()-1:  # if it's the last call of __getitem__ create the combined file
                np.save(f'{self.tableName}_all_indices.npy', self.indices)
                np.save(f'{self.tableName}_{self.num_sig_f}_signals.npy', self.signals)
                np.save(f'{self.tableName}_all_labels.npy', self.labels)
                os.remove(f'{self.tableName}_{self.num_sig_f - 1}_indices.npy')  # deleting the shorter ver of the file
                os.remove(f'{self.tableName}_{self.num_sig_f - 1}_labels.npy')

                self.iterator2 = 0  # repurposing this variable
                while os.path.exists(f'{self.tableName}_{self.iterator2}_signals.npy'):  # combining files into one
                    with open(f'{self.tableName}_{self.iterator2}_signals.npy', 'rb') as f:
                        if self.iterator2 == 0:
                            x = np.load(f, allow_pickle=True)
                        else:
                            x = np.concatenate((x, np.load(f, allow_pickle=True)))
                    self.iterator2 += 1
                shape = x.shape
                final = np.zeros((2, int(shape[0] / 2), shape[1]))  # shape: two channels X number of segments (=samples) X num. of voltage recordings (e.g. 1000)
                final[0] = x[0:shape[0]:2, :]  # take one in two segments. Odd segments are one channel and even segments are the other one
                final[1] = x[1:shape[0]:2, :]
                final = np.delete(final, np.s_[1000::1000],
                                  1)  # deleting duplicate EEG segments. One in every 1000 with the first deletion at index 1000
                np.save(f'{self.tableName}_all_signals.npy', final)
                self.signals = torch.tensor(final)
                del final
                self.iterator2 = 0
                while os.path.exists(f'{self.tableName}_{self.iterator2}_signals.npy'):  # deleting redundant files
                    os.remove(f'{self.tableName}_{self.iterator2}_signals.npy')
                    self.iterator2 += 1

    def __len__(self):
        return len(self.table)  # database size


def recover_dls(eeg_ds, old_dir_path, batch_size=128, num_workers=64):
    groups = ('Train', 'Test', 'Validation')
    for dl_i in range(3):
        samp_indices = pd.read_csv(old_dir_path + "dl_" + groups[dl_i] + ".csv").to_numpy()
        samp_indices = samp_indices[:, 1]
        if dl_i == 0:
            dl_train = DataLoader(eeg_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                  sampler=SubsetRandomSampler(np.random.permutation(samp_indices)))
        elif dl_i == 1:
            dl_test = DataLoader(eeg_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                 sampler=SubsetRandomSampler(np.random.permutation(samp_indices)))
        elif dl_i == 2:
            dl_val = DataLoader(eeg_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                                sampler=SubsetRandomSampler(np.random.permutation(samp_indices)))
    return dl_train, dl_test, dl_val
