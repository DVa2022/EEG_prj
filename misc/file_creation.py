import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.datasets import sleep_physionet
from mne.time_frequency import psd_welch
import datetime


# this is how you access fif files:
# tr = mne.io.read_raw_fif(fname='4041E0-PSG_raw.fif')
# print(tr[0][0][0][0:4]) # data from the first channel
# print(tr[1][0][0][0:4]) # data from the second channel
# print(tr[0][1][10:15]) # this is how you access the time in seconds


def segment_hours(subjectNum, recordingNum):
    # this function creates a csv file which contains the start and end hours of each segment
    paths = fetch_data(subjects=[subjectNum], recording=[recordingNum])
    raw_train = mne.io.read_raw_edf(paths[0][0], stim_channel='Event marker', misc=['Temp rectal'])
    raw_train.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])
    annot_train = mne.read_annotations(paths[0][1])
    raw_train.set_annotations(annot_train, emit_warning=False)
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    annot_train.crop(annot_train[1]['onset'] - 60 * 3, annot_train[-2]['onset'] + 30)
    raw_train.set_annotations(annot_train, emit_warning=False)
    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)


    elapsed_seconds = events_train[:, 0] / 100  # dividing the sample indeces by 100 to get the second numbers

    beginning = raw_train.info['meas_date']
    time_in_seconds = beginning.hour * 60 * 60 + beginning.minute * 60 + beginning.second + elapsed_seconds  # the seconds passed since 00:00:00
    time_in_seconds[time_in_seconds >= 24 * 60 * 60] = time_in_seconds[time_in_seconds >= 24 * 60 * 60] - 24 * 60 * 60
    time_in_seconds_shifted_by_30 = time_in_seconds + 30
    time_in_seconds_shifted_by_30[time_in_seconds_shifted_by_30 >= 24 * 60 * 60] = time_in_seconds_shifted_by_30[
                                                                                       time_in_seconds_shifted_by_30 >= 24 * 60 * 60] - 24 * 60 * 60

    epoch_start = np.array([str(datetime.timedelta(seconds=s)) for s in time_in_seconds])  # an array of many HH:MM:SS
    epoch_end = np.array([str(datetime.timedelta(seconds=s)) for s in time_in_seconds_shifted_by_30])
    epoch_times = np.vstack((epoch_start, epoch_end)).T

    np.savetxt("segmentTimes.csv", epoch_times, delimiter=",", fmt='%8s')

def getPaths():
    # the function creates a file containing the paths of all the recordings with the same order like in 'simple table
    # getPaths is not the smartest written function but it works
    i = 0

    while i < 200:
        try:
            paths.append(fetch_data(subjects=[i], recording=[1]))
        except:
            i = i
        try:
            paths.append(fetch_data(subjects=[i], recording=[2]))
        except:
            i = i
        i = i + 1

    i = 0
    while i < 100:
        try:
            paths.append(mne.datasets.sleep_physionet.temazepam.fetch_data(subjects=[i]))
        except:
            i = i
        try:
            paths.append(mne.datasets.sleep_physionet.temazepam.fetch_data(subjects=[i]))
        except:
            i = i
        i = i + 1

    return paths


def main():
    # the following code creates filtered fif files from the original recordings
    paths = getPaths()
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    RecordingNumbers = [x for x in range(0, 198)]

    for rec in RecordingNumbers:
        try:
            raw = mne.io.read_raw_edf(paths[rec][0], stim_channel='Event marker', misc=['Temp rectal'])
            path = paths[rec][0]
        except:
            raw = mne.io.read_raw_edf(paths[rec][0][0], stim_channel='Event marker', misc=['Temp rectal'])
            path = paths[rec][0][0]

        new_file_name = path[-16:-4] + '_raw.fif'
        new_events_file_name = path[-16:-4] + '-eve.fif'

        raw.pick_channels(channels)

        try:
            annot_train = mne.read_annotations(paths[rec][1])
        except:
            annot_train = mne.read_annotations(paths[rec][0][1])

        raw.set_annotations(annot_train, emit_warning=False)

        annotation_desc_2_event_id = {'Sleep stage W': 1,
                                      'Sleep stage 1': 2,
                                      'Sleep stage 2': 3,
                                      'Sleep stage 3': 4,
                                      'Sleep stage 4': 4,
                                      'Sleep stage R': 5}

        # if we wish to crop the recording, so we're left only with the sleep we should use these lines:
        # annot_train.crop(annot_train[1]['onset'] - 60*3, annot_train[-2]['onset'] + 30)
        # try:
        #     raw.crop(tmin = annot_train[1]['onset'] - 60*3, tmax = annot_train[-2]['onset'] + 30)
        # except: # this exception is relevant only for patient 19, if I'm not mistaken
        #     raw.crop(tmin = annot_train[1]['onset'] - 60*3, tmax = annot_train[-2]['onset'] + 30 - 0.01)

        raw.load_data()
        raw = raw.filter(l_freq=1, h_freq=49)

        raw.save(new_file_name,overwrite=True)

        events_train, _ = mne.events_from_annotations(
            raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        event_id = {'Sleep stage W': 1,
                    'Sleep stage 1': 2,
                    'Sleep stage 2': 3,
                    'Sleep stage 3/4': 4,
                    'Sleep stage R': 5
                    }
        tmax = 30. - 1. / raw.info['sfreq']
        print('events shape:', events_train.shape)
        # if we wish to create events files we can use this:
        # mne.write_events(new_events_file_name, events_train, overwrite=True)
        # events files contain the sample number of each event and its annotation (between 1 to 5)

        # if we wish to create epochs we should notice that there's an error when we use the next line of code.
        # epochs_train = mne.Epochs(raw=raw, events=events_train, event_id=event_id, tmin=0., tmax=tmax, baseline=None)
        # the error is: ValueError: No matching events found for Sleep stage 3/4 (event id 4)
        # we get it only for recording 2 of patient 20

        # if we wish to save the epochs we should use the next line:
        # epochs_train.save(epochsFileName)
        # we should notice that there's an error with it, I think.

        del raw