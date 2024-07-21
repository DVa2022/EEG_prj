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
from time import time

def build_simple_table():
    """
    Builds table for experiment x
    Experiment description:
    Builds "simple table.csv"
    """
    # if output is saved does not rerun
    try:
        simple_tbl = pd.read_csv("simple table.csv", header=0, index_col=0)
        print("{} Already exists".format("simple table.csv"))
        return simple_tbl
    except:
        pass
    print("{} did not already exist".format("simple table.csv"))
    print("creating now")
    # removed need for input
    if 'C:\\Users\\atver' in os.getcwd():  # is my cpu or is not ie triton
        table_paths = ("C:\\Users\\atver\\mne_data\\physionet-sleep-data\\SC-subjects.xls","C:\\Users\\atver\\mne_data\\physionet-sleep-data\\ST-subjects.xls")
    else:
        table_paths = ("\\home\\stu16\\physionet_data\\SC-subjects.xls","\\home\\stu16\\physionet_data\\ST-subjects.xls")
    # Build simple_table ###
    sctable_path, sttable_path = table_paths
    sctable = pd.read_excel(sctable_path, sheet_name=0, header=0)
    sttable = pd.read_excel(sttable_path, sheet_name=0, header=1)
    # combine both table data into one panda SC then below ST placebo then ST temazepan
    # using column orginzation from SC
    sub = sttable['Nr'] + 82  # increase index so that ST patient index follows SC index
    nightp = sttable['night nr']
    age = sttable['Age']
    sex = (- (sttable['M1/F2']-1)) +2  # match convention in SC F=1 M=2
    offp = sttable['lights off']
    nightt = sttable['night nr.1']
    offt = sttable['lights off.1']
    night1 = pd.concat([sub, nightp, age, sex, offp], axis=1)
    night1.columns = sctable.columns
    night2 = pd.concat([sub, nightt, age, sex, offt], axis=1)
    night2.columns = sctable.columns
    # ignore_index stops the row index from restarting
    simple_table = pd.concat([sctable,night1,night2],ignore_index =True)
    # save the built table
    simple_table.to_csv("simple table.csv")
    return simple_table # table of segments


class PathFinder:
    def __init__(self, filtered, tbl):
        # filtered: a flag to choose either original data (False) or filtered data (True)
        # tbl: the table to use for looking up the recordings. uses the index in the
        #      table and the corresponding pid and night that appear there
        self.tbl = tbl
        # set the path to the data
        filtered =0 #TODO hard set
        self.f = filtered
        if filtered:
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
        # index: index of recording as appearing in simple_table
        db = self.DB_path
        # s_tbl_path = 'simple table.csv'
        # tbl = pd.read_csv(s_tbl_path, index_col=0)

        ss = self.tbl['subject'][index]
        n = self.tbl['night'][index]
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
        leads = db + leads
        annotations = db + annotations
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

    def __call__(self, signal, rosh=-1, sof=-1):
        """
        params:
              rosh: int
                  index of time series data where segment begins. if not provided use definition in function
              sof: int
                  index of time series data where segment ends. if not provided use definition in function
        """
        # ------Your code------#
        # change assume subsegmenting and normalization before transform

        # print('djskh',signal.max(),signal.min())
        # Transform the data type from double (float64) to single (float32) to match the later network weights.
        t_signal = signal.astype(np.single)
        # We transpose the signal to later use the lead dim as the channel... (C,L).
        t_signal = torch.reshape(torch.tensor(t_signal), (t_signal.size, 1))
        # TODO Eran used t_signal = torch.unsqueeze(t_signal, dim=1)   # Add dim as expected in the RNN layer class, instead of torch.transpose(torch.tensor(t_signal), 0, 1) Needs it because he used only one lead
        # ------^^^^^^^^^------#
        return t_signal  # Make sure I am a PyTorch Tensor


class EEGDataset(Dataset):

    def __init__(self, Transform, exp_table, filtered=0): #TODO changed to see effect on loading
        super().__init__()  # When using a subclass, remember to inherit its properties.
        # ------Your code------#
        # Define self.EEG_path, self.exp_table (pandas from build experiment)
        # and self.transform (create an object from the transform we implemented):
        if 'C:/Users/atver' in os.getcwd():  # is my cpu
            self.EEG_path = "C:/Users/atver/mne_data/filt_data/"  # getting data from filtered data
        else:  # or is not ie triton
            self.EEG_path = "/home/stu16/filt_data/"
        self.table = exp_table
        self.transform = Transform()
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

        # when called on index gives the file paths for signal and annotations
        self.get_rec_paths = PathFinder(filtered=filtered, tbl=self.table).get_paths
        self.asked_simple = 0 # did I already ask if this is simple table
        # ------^^^^^^^^^------#


    def get_label(self, index):
        # A method to decide the label:
        age = self.table['age'][index]  # TODO return age label as well
        sex = self.table['sex (F=1)'][index] - 1  # F1M2 switch to F0M1
        return sex, age  # F0M1

    def __getitem__(self, index):
        # ------Your code------# could double samples to examin by making if on odd even indexes to switch between the two channels
        # Read the record with mne and transform its signal. Assign a label by using get_label.
        index = int(index)
        # Get data from specified recording
        rec_paths = self.get_rec_paths(index)
        # start = time()
        with contextlib.redirect_stdout(io.StringIO()) as f:
            if 'edf' in rec_paths[0]:
                raw = mne.io.read_raw_edf(rec_paths[0])
            elif 'fif' in rec_paths[0]:
                raw = mne.io.read_raw_fif(rec_paths[0])
            meas_date = raw.info['meas_date']
        # print('measdd' ,meas_date)
        # Get rosh and sof if in table
        # print("running time:", time()-start)
        try:
            # print('dgs',self.table['srt'][index])
            rosh_arr = np.array([float(x) for x in self.table['srt'][index].split(':')])
            rosh_sec = (rosh_arr * np.array([3600, 60, 1])).sum()
            meas_sec = (meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second)
            # print('rosh_sec',rosh_sec)
            # print('meas_sec',meas_sec)
            if meas_sec > rosh_sec:
                rosh = (rosh_sec + (24 * 3600 - meas_sec))
            else:
                rosh = (rosh_sec - meas_sec)
            try:
                sof = rosh + self.seg_len - .01  # seg_len is inferred from tabular data minus .01 because going by time not index
            except:
                sof = rosh + 0.5 - .01

        except (NameError, KeyError):  # error is rosh is not defined
            if not self.asked_simple:
                self.asked_simple += 1
                print('is this simple path ?')  # should only see if there are no segment times specified
            rosh = 3600
            sof = 3600 + 30 - .01
            pass

            # crop by rosh and sof now to save memory. raw.crop uses seconds not index
        raw.crop(rosh, sof)

        # Apply the transform to get the right shape and data type
        # raw.get_data(picks='MEG 0113', start=1000, stop=2000, return_times=True) #example line from mne
        # channels are tuples with ind0 containing the signal and ind1 holding the times in seconds
        # channel names     ['EEG Fpz-Cz', 'EEG Pz-Oz']
        fpzczt = self.transform(raw.get_data(picks='EEG Fpz-Cz', return_times=False))
        signal = np.zeros((2, len(fpzczt)))  # use transformed lead length to make signal array #zero divison concern?
        signal[0, :] = fpzczt.squeeze()
        signal[1, :] = (self.transform(raw.get_data(picks='EEG Pz-Oz',
                                                    return_times=False))).squeeze()  # get and transform same line to save memory

        label = self.get_label(index)
        signal = torch.tensor(signal)
        signal = signal.to(dtype=torch.double)
        # ------^^^^^^^^^------#
        return signal, label

    def __len__(self):
        return len(self.table)  # database size


def filter_save(overwrite):
    # the following code creates filtered fif files from the original recordings
    newDBpath = 'C:/Users/atver/mne_data/filt_data/'
    # newDBpath = 'home/stu16/filt_data/'
    tbl = build_simple_table()
    get_paths = PathFinder(filtered=False, tbl=tbl).get_paths
    for rec in range(len(tbl)):
        orig_path, orig_annot_path = get_paths(rec)
        new_file_path = newDBpath + orig_path[-16:-4] + '_raw.fif'
        new_events_file_path = newDBpath + orig_path[-16:-4] + '-eve.fif'
        if not overwrite and os.path.exists(new_file_path) and os.path.exists(new_events_file_path):
            continue
        with contextlib.redirect_stdout(io.StringIO()) as f:
            raw = mne.io.read_raw_edf(orig_path, stim_channel='Event marker', misc=['Temp rectal'])
        ch = ['EEG Fpz-Cz', 'EEG Pz-Oz']
        raw.pick_channels(ch, ordered=True)
        # filtering freq
        with contextlib.redirect_stdout(io.StringIO()) as f:
            raw.load_data()
            raw = raw.filter(l_freq=1, h_freq=49)

            # for all files the first time pt is zero so I don't need to copy the time vector
        # print('sdsa ifrst timept',raw[ch[0]][1][0])

        # Normalize
        datal = raw[ch[0]][0]
        # print((datal.shape))
        datalmin = datal.min()
        data = np.zeros((2, datal.shape[1]))
        data[0, :] = (datal - datalmin) / (datal.max() - datalmin)
        datal = raw[ch[1]][0]
        datalmin = datal.min()
        data[1, :] = (datal - datalmin) / (datal.max() - datalmin)
        with contextlib.redirect_stdout(io.StringIO()) as f:
            raw = mne.io.RawArray(data, raw.info, first_samp=0, copy='auto', verbose=None)
        annot_train = mne.read_annotations(orig_annot_path)
        raw.set_annotations(annot_train, emit_warning=False)

        # save the data after the filtering
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                raw.save(new_file_path, overwrite=overwrite)
        except FileExistsError:
            pass
        try:
            annot_train.save(new_events_file_path, overwrite=overwrite)
        except FileExistsError:
            pass


def segment_hours(tbl_fn, seg_len=30, stages={'Sleep stage 1': 2}, limit=False):
    # seg_len: segment length in seconds
    # stages is dictionary of sleep stages to use
    # tbl_fn: file name of the table
    # this function creates a csv file which contains the start and end hours of each segment
    s_tbl_path = 'simple table.csv'
    tbl = pd.read_csv(s_tbl_path, header=0,index_col=0)
    new_tbl = pd.DataFrame(columns=['subject','night','age','sex (F=1)','LightsOff'])
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
        with contextlib.redirect_stdout(io.StringIO()) as f:
            events_train, _ = mne.events_from_annotations(raw0, event_id=stages, chunk_duration=seg_len)
        print('events_train shape : ', events_train.shape)
        # if a limit was chosen shuffle the found segments and choose only that many

        if (events_train.shape[0] > limit) and limit:  # use all segments if no. segments found is less than the limit
            np.random.shuffle(events_train)  # acts in place
            # print('341341', event_onsets)
            events_train = events_train[0:limit, :]
        # dividing the sample indices by 100 to get the second numbers is
        event_onsets = events_train[:, 0] / 100   # np.ndarray
        # should be equal to limit now
        # print('3432', event_onsets.shape[0])
        if event_onsets.shape[0] != limit:
            print(paths[0])
            print('event_onsets shape: ', event_onsets.shape)
            print('events_train: ',events_train)
        meas_date = raw0.info['meas_date']
        time_in_seconds = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second + event_onsets  # the seconds passed since 00:00:00
        time_in_seconds[time_in_seconds >= 24 * 3600] = time_in_seconds[time_in_seconds >= 24 * 3600] - 24 * 3600

        time_in_seconds_end = time_in_seconds + seg_len
        time_in_seconds_end[time_in_seconds_end >= 24 * 3600] = time_in_seconds_end[ time_in_seconds_end >= 24 * 3600] - 24 * 3600

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
    experiments = {"sleep_1_10sec.csv": {'tbl_fn': "sleep_1_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage 1': 2}},
                   "sleep_1_30sec.csv": {'tbl_fn': "sleep_1_30sec.csv", 'seg_len': 30, 'stages': {'Sleep stage 1': 2}},
                   "sleep_1_45sec.csv": {'tbl_fn': "sleep_1_45sec.csv", 'seg_len': 45, 'stages': {'Sleep stage 1': 2}},
                   "sleep_1_60sec.csv": {'tbl_fn': "sleep_1_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage 1': 2}},
                   "sleep_1_120sec.csv": {'tbl_fn': "sleep_1_120sec.csv", 'seg_len': 120, 'stages': {'Sleep stage 1': 2}},
                   "sleep_1_240sec.csv": {'tbl_fn': "sleep_1_2400sec.csv", 'seg_len': 240, 'stages':{'Sleep stage 1': 2}},
                   "sleep_1_480sec.csv": {'tbl_fn': "sleep_1_4800sec.csv", 'seg_len': 480, 'stages':{'Sleep stage 1': 2}},
                   "sleep_2_10sec.csv": {'tbl_fn': "sleep_2_10sec.csv", 'seg_len': 10, 'stages': {'Sleep stage 2': 2}},
                   "sleep_2_30sec.csv": {'tbl_fn': "sleep_2_30sec.csv", 'seg_len': 30, 'stages': {'Sleep stage 2': 2}},
                   "sleep_2_45sec.csv": {'tbl_fn': "sleep_2_45sec.csv", 'seg_len': 45, 'stages': {'Sleep stage 2': 2}},
                   "sleep_2_60sec.csv": {'tbl_fn': "sleep_2_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage 2': 2}},
                   "sleep_2_120sec.csv": {'tbl_fn': "sleep_2_120sec.csv", 'seg_len': 120, 'stages': {'Sleep stage 2': 2}},
                   "sleep_2_240sec.csv": {'tbl_fn': "sleep_2_240sec.csv", 'seg_len': 240, 'stages': {'Sleep stage 2': 2}},
                   "sleep_2_480sec.csv": {'tbl_fn': "sleep_2_480sec.csv", 'seg_len': 480, 'stages': {'Sleep stage 2': 2}},
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
                   "sleep_123_60sec.csv": {'tbl_fn': "sleep_123_60sec.csv", 'seg_len': 60, 'stages': {'Sleep stage 1': 1, 'Sleep stage 2': 2,'Sleep stage 3':3}},
                   "sleep_1_30sec_lim5.csv": {'tbl_fn': "sleep_1_30sec_lim5.csv", 'seg_len': 30, 'stages': {'Sleep stage 1': 2}, 'limit': 5},
                   "sleep_1_10sec_lim15.csv": {'tbl_fn': "sleep_1_10sec_lim15.csv",  'seg_len': 10, 'stages': {'Sleep stage 1': 2},'limit': 15},
                   "sleep_1_2sec_lim75.csv": {'tbl_fn': "sleep_1_2sec_lim75.csv", 'seg_len': 2, 'stages': {'Sleep stage 1': 2}, 'limit': 75},
                   "sleep_1_2sec_lim10.csv": {'tbl_fn': "sleep_1_2sec_lim10.csv", 'seg_len': 2, 'stages': {'Sleep stage 1': 2}, 'limit': 10},
                   "sleep_1_5sec_lim10.csv": {'tbl_fn': "sleep_1_5sec_lim10.csv", 'seg_len': 5, 'stages': {'Sleep stage 1': 2}, 'limit': 10},
                   "sleep_1_5sec_lim5.csv": {'tbl_fn': "sleep_1_5sec_lim5.csv",  'seg_len': 5, 'stages': {'Sleep stage 1': 2},'limit': 5},
                   "sleep_1_30sec_lim180.csv": {'tbl_fn': "sleep_1_30sec_lim180.csv",  'seg_len': 30, 'stages': {'Sleep stage 1': 2}, 'limit': 180},
                   "sleep_2_30sec_lim180.csv": {'tbl_fn': "sleep_2_30sec_lim180.csv", 'seg_len': 30, 'stages': {'Sleep stage 2': 2}, 'limit': 180},
                   "sleep_2_05sec_lim180.csv": {'tbl_fn': "sleep_2_05sec_lim180.csv", 'seg_len': 0.5, 'stages': {'Sleep stage 2': 2}, 'limit': 180},
                   "sleep_2_30sec_lim90.csv": {'tbl_fn': "sleep_2_30sec_lim90.csv",  'seg_len': 30, 'stages': {'Sleep stage 2': 2}, 'limit': 90},
                   "sleep_1_05sec_lim1.csv": {'tbl_fn': "sleep_1_05sec_lim1.csv",  'seg_len': 0.5, 'stages': {'Sleep stage 1': 2}, 'limit': 1},
                   "sleep_2_05sec_lim1.csv": {'tbl_fn': "sleep_2_05sec_lim1.csv",  'seg_len': 0.5, 'stages': {'Sleep stage 2': 2}, 'limit': 1},
                   "sleep_2_05sec_lim3.csv": {'tbl_fn': "sleep_2_05sec_lim3.csv",  'seg_len': 0.5, 'stages': {'Sleep stage 2': 2}, 'limit': 3}}

    assert(experiment_fn == experiments[experiment_fn]['tbl_fn'])
    new_tbl = segment_hours(**experiments[experiment_fn])
    return new_tbl



def tt_split_by_pid_mf(dataset, batch_size, train_rt=.8,  num_workers=0, verbose=1):
    """
    Finds a split within 2 percent of split specified that keeps patients in test set out of train set.
    :param dataset: PyTorch dataset to split.
    :param train_rt: Split ratio for the data - e.g., 0.8 ==> 80% train and 20% spilt between test and validation
    :param batch_size: Define the batch_size to use in the DataLoaders.
    :param num_workers: Define the num_workers to use in the DataLoaders.
    :return: train and test DataLoaders.
    """
    # ------Your code------#
    # since some patients have less data must verify that this split is adequate
    acptE = 0.05  # 1 percent
    error_tr = nf = 1
    iteration = nm = 0
    pid_samp = dataset.table['subject']  # patient id for each sample as in table
    while np.logical_or(error_tr > acptE, nf != nm):
        u_pids = np.unique(pid_samp)  # list of unique patient ids
        npt = len(u_pids)
        # change number on different iterations for more randomness and higher chance of finding a good split
        n_test_pt = int(round(npt*train_rt)+np.random.randint(-3, 3, size=(1,), dtype=int))
        # ensures that pts with only one recording are not always in the test set
        random_order = np.random.permutation(u_pids)
        train_pts = random_order[:n_test_pt]
        test_pts = random_order[n_test_pt:]
        # use test pts to get indexes of samples for those patients, shuffle them, and pass them to a data loader
        # should only be one d # need the first index because it was returning a tuple now ndarray(number,)
        test_samp_indexes = np.where(np.isin(pid_samp, test_pts))[0]
        dl_test = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(test_samp_indexes)), num_workers=num_workers)
        # train
        train_samp_indexes = np.where(np.isin(pid_samp, train_pts))[0]  # should only be one d
        dl_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(train_samp_indexes)), num_workers=num_workers)
        # calculate the split ratios for evaluation
        actual_tr_split = len(dl_train.sampler.indices)/dataset.__len__()
        error_tr = abs(train_rt - actual_tr_split)  # TODO
        nf = (dataset.table['sex (F=1)'][train_samp_indexes] == 1).sum()
        nm = (dataset.table['sex (F=1)'][train_samp_indexes] == 2).sum()
        # since some patients have less data must verify that this split is adequate
        iteration += 1
        if iteration == 1500:
            print("could not find a good split after 1500 tries")
            print("error tr = {:.4f}".format(error_tr))
            break
        if verbose > 1:
            print('iteration', iteration)
            print('dl_train', len(dl_train.sampler.indices))
            print('dl_test', len(dl_test.sampler.indices))
            print('actual tr split {:.4f}'.format(actual_tr_split))
            print('actual tst split {:.4f}'.format(len(dl_test.sampler.indices)/dataset.__len__()))
    if verbose:
        print('n_pts dl_train', len(dl_train.sampler.indices))
        print('n_pts dl_test', len(dl_test.sampler.indices))
        print('actual tr split {:.4f}'.format(actual_tr_split))
        print('actual tst split {:.4f}'.format(len(dl_test.sampler.indices)/dataset.__len__()))
    # ------^^^^^^^^^------#
    return dl_train, dl_test, dl_test # TODO I did this on purpose

def train_test_split_naive(dataset, ratio, batch_size, num_workers=0):
    """  for reference only
    :param dataset: PyTorch dataset to split.
    :param ratio: Split ratio for the data - e.g., 0.8 ==> 80% train and 20% test
    :param batch_size: Define the batch_size to use in the DataLoaders.
    :param num_workers: Define the num_workers to use in the DataLoaders.
    :return: train and test DataLoaders.
    """
    # ------Your code------#
    # Hint: You can use torch.randperm to shuffle the data before applying the split.
    length = dataset.__len__()
    random_order = torch.randperm(length)
    N_test = int(round(length*ratio))
    dl_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(random_order[:N_test]), num_workers=num_workers)
    dl_test = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(random_order[N_test:]), num_workers=num_workers)
    # ------^^^^^^^^^------#
    return dl_train, dl_test


def train_test_split_by_pid(dataset, ratio, batch_size, num_workers=0):
    """
    Finds a split within 2 percent of split specified that keeps patients in test set out of train set.
    No need to use just reference
    :param dataset: PyTorch dataset to split.
    :param ratio: Split ratio for the data - e.g., 0.8 ==> 80% train and 20% test
    :param batch_size: Define the batch_size to use in the DataLoaders.
    :param num_workers: Define the num_workers to use in the DataLoaders.
    :return: train and test DataLoaders.
    """
    # ------Your code------#
    # since some patients have less data must verify that this split is adequate
    acptE = 0.0025 # .5 percent
    error = 1
    iter = 0
    pid_samp = dataset.table['subject'] # patient id for each sample as in table
    while error > acptE:
        u_pids = np.unique(pid_samp)  # list of unique patient ids
        Npt = len(u_pids)
        N_testpt = int(round(Npt*ratio)) # could change number on different iterations for more randomness and chance of finding a good split
        random_order = np.random.permutation(u_pids)
        trainpts = random_order[:N_testpt]
        testpts = random_order[N_testpt:]
        # use test pts to get indexes of samples for those patients, shuffle them, and pass them to a data loader
        test_samp_indexes = np.where(np.isin(pid_samp,testpts))[0] # should only be one d # need the first index because it was returning a tuple now ndarray(number,)
        dl_test = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(test_samp_indexes)), num_workers=num_workers)
        print('dl_test',len(dl_test.sampler.indices))
        train_samp_indexes = np.where(np.isin(pid_samp,trainpts))[0] # should only be one d
        dl_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(train_samp_indexes)), num_workers=num_workers)
        actual_split = len(dl_train.sampler.indices)/dataset.__len__()
        error = abs(ratio - actual_split )
        # since some patients have less data must verify that this split is adequate
        iter +=1
        if (iter==100):
            print("could not find a good split after 100 tries")
            print("error = {:.4f}".format(error))
            break
    print('actual split {:.4f}'.format(actual_split))
    # ------^^^^^^^^^------#
    return dl_train, dl_test


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
    while np.logical_or(error_tr > acptE, error_vl > acptE):
        u_pids = np.unique(pid_samp)  # list of unique patient ids
        npt = len(u_pids)
        # change number on different iterations for more randomness and higher chance of finding a good split
        n_test_pt = int(round(npt*train_rt)+np.random.randint(-3, 3, size=(1,), dtype=int))
        # ensures that pts with only one recording are not always in the test set
        n_val_pt = int(round(npt*val_rt)+np.random.randint(-3, 3, size=(1,), dtype=int))
        random_order = np.random.permutation(u_pids)
        train_pts = random_order[:n_test_pt]
        val_pts = random_order[n_test_pt:(n_test_pt+n_val_pt)]
        test_pts = random_order[(n_test_pt+n_val_pt):]
        # use test pts to get indexes of samples for those patients, shuffle them, and pass them to a data loader
        # should only be one d # need the first index because it was returning a tuple now ndarray(number,)
        test_samp_indexes = np.where(np.isin(pid_samp, test_pts))[0]
        dl_test = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(test_samp_indexes)), num_workers=num_workers)
        # validation
        # should only be one d # need the first index because it was returning a tuple now ndarray(number,)
        val_samp_indexes = np.where(np.isin(pid_samp, val_pts))[0]
        dl_val = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(val_samp_indexes)), num_workers=num_workers)
        # train
        train_samp_indexes = np.where(np.isin(pid_samp, train_pts))[0]  # should only be one d
        dl_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(np.random.permutation(train_samp_indexes)), num_workers=num_workers)
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
            print('dl_val', len(dl_val.sampler.indices))
            print('dl_test', len(dl_test.sampler.indices))
            print('actual tr split {:.4f}'.format(actual_tr_split))
            print('actual vl split {:.4f}'.format(actual_vl_split))
            print('actual tst split {:.4f}'.format(len(dl_test.sampler.indices)/dataset.__len__()))
    if verbose:
        print('n_pts dl_train', len(dl_train.sampler.indices))
        print('n_pts dl_val', len(dl_val.sampler.indices))
        print('n_pts dl_test', len(dl_test.sampler.indices))
        print('actual tr split {:.4f}'.format(actual_tr_split))
        print('actual vl split {:.4f}'.format(actual_vl_split))
        print('actual tst split {:.4f}'.format(len(dl_test.sampler.indices)/dataset.__len__()))
    # ------^^^^^^^^^------#
    return dl_train, dl_val, dl_test


class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Define self.direct_path by adding the layers into a nn.Sequential. Use nn.Conv1d and nn.Relu.
        # You can use padding to avoid reducing L size, to allow the skip-connection adding.
        # ------Your code------#
        self.direct_path = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        )
        # ------^^^^^^^^^------#

        # Define self.skip_layers path.
        # You should use convolution layer with a kernel size of 1 to consider the case where the input and output shapes mismatch.
        # ------Your code------#
        skip_layers = []
        if in_channels != 32:  # HOW DOES THIS PART WORK? When are you adding the layers
            skip_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1,
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


# Training loop
def forward_epoch(model, dl, loss_function, optimizer, weight, total_loss=0,
                  to_train=False, desc=None, device=torch.device('cpu'), label=0):
    # label =0 is for sex
    # label = 1 is for Age
    # total loss is over the entire epoch
    # y_trues is by patient for the entire epoch; can get last batch with [-batch_size]
    # y_preds is by patient for the entire epoch
    #
    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        model = model.double().to(device)  # solving runtime memory issue

        y_trues = torch.empty(0).type(torch.int).to(device)
        y_preds = torch.empty(0).type(torch.int).to(device)
        for i_batch, (X, y) in enumerate(dl):
            # print('sickyall',X.dtype)
            X = X.to(device)
            X = X.type(torch.double)
            # print('wackyall',X.dtype)
            y = y[label].to(device)  # added index because of get label returning sex, age

            # Forward:
            # print(X.shape)
            try:
                y_pred, aa = model(X)
                # didn't work:
                # y_pred = torch.mean(y_pred,1)[:,0]
                y_pred = y_pred[:, 1, 0]
            except:
                y_pred = model(X)
            # Loss:
            y_true = y.type(torch.double)
            y_true_copy = torch.clone(y_true)
            # weightsForLoss = weight*y_true_copy  # TODO y_true=[1, 0, 0, 1, 0] weightsForLoss=[1.5, 1, 1, 1.5, 1]  y_pred=[0.12, 0.34, 0.11, 0.64, 0.22]
            # weightsForLoss = weightsForLoss.detach()
            # loss_func = loss_function(weightsForLoss)
            loss = loss_function(y_pred, y_true) # loss of one batch
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



def name_dir(run_desc):
    # only will work on server
    # get last named dir
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



