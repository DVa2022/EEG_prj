# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.data import colorwheel as colourwheel
from typing import Tuple, List

def build_table(table_paths):
    """
    Builds table for experiment
    saves table to "simple samp.xls"
    """
    import pandas as pd
    save_name = "simple table.csv"
    # if output is saved does not rerun
    try:
        combtable = pd.read_csv(save_name, header=0)
        message = "{} Already exists".format(save_name)
        print(message)
        return None
    except:
        pass
    print("{} did not already exist".format(save_name))
    print("creating now")

    ### Build simple_table ###
    sctable_path, sttable_path = table_paths
    sctable = pd.read_excel(sctable_path, sheet_name=0, header=0)
    sttable = pd.read_excel(sttable_path, sheet_name=0, header=1)  ############# why 1?
    # combine both table data into one panda SC then below ST placebo then ST temazepan
    # using column organization from SC
    sub = sttable['Nr'] + 82  # increase index so that ST patient index follows SC index
    nightp = sttable['night nr']
    age = sttable['Age']
    sex = (- (sttable['M1/F2'] - 1)) + 2  # match convention in SC F=1 M=2
    offp = sttable['lights off']
    nightt = sttable['night nr.1']
    offt = sttable['lights off.1']
    night1 = pd.concat([sub, nightp, age, sex, offp], axis=1)
    night1.columns = sctable.columns
    night2 = pd.concat([sub, nightt, age, sex, offt], axis=1)
    night2.columns = sctable.columns
    simple_table = pd.concat([sctable, night1, night2], ignore_index=True)  # the row index is restarting
    ######

    # save the built table
    simple_table.to_csv(save_name)
    return simple_table


def max_lloyd_iter(img: np.ndarray, prev_levels: np.ndarray) -> np.ndarray:
    '''

    Add your description and complete the inputs (params) and output (return).

    :param img: a 3D numpy array representing an image with at least one channel
    :param prev_levels: a ****************** numpy array representing the previous quantization levels
    :return new_levels: a ****************** numpy array representing the new quantization levels
    '''
    # ====== YOUR CODE: ======
    new_levels = np.zeros(shape=prev_levels.shape, dtype='float32')
    numOfChannels = prev_levels.shape[1]
    level_num = prev_levels.shape[0]
    # print('prev_levels',prev_levels,prev_levels.shape)
    prev_levels = np.reshape(prev_levels, (level_num, 1, 1, numOfChannels))  # preparing it for the next line
    # print('prev_levels',prev_levels,prev_levels.shape)
    distances = np.linalg.norm(img - prev_levels,
                               axis=3)  # getting the distances of all the quantization levels to all pixels.
    #  We get an array with the size of number of quantization levels X image height X image width
    # print('distances',distances)
    indices_of_minima = np.argmin(distances,
                                  axis=0)  # an array with the shape of the image. Each element in the array contains the
    # index of the closest quantization level to the pixel in the same place
    # print('indices_of_minima',indices_of_minima)
    print("bbbbb", prev_levels.shape[0])

    for ind in range(0, prev_levels.shape[0]):
        # print('ind:\n', ind)
        # print('indices_of_minima\n',indices_of_minima)
        # print('indices_of_minima==ind\n',indices_of_minima==ind)
        # print('np.any(indices_of_minima==ind)\n',np.any(indices_of_minima==ind))
        # print('img[indices_of_minima==ind]\n',img[indices_of_minima==ind])
        # print('np.mean(img[indices_of_minima==ind],axis=0)\n',np.mean(img[indices_of_minima==ind],axis=0))
        if np.any(indices_of_minima == ind):  # if the quantization level has at least one pixel which is closest to it
            print("np.any(indices_of_minima==ind)", np.any(indices_of_minima == ind))
            print("new_levels[ind]", new_levels[ind])
            # print("np.mean(img[indices_of_minima==ind],axis=0)",np.mean(img[indices_of_minima==ind],axis=0))
            new_levels[ind] = np.mean(img[indices_of_minima == ind], dtype='float32',
                                      axis=0)  # calculating the mean of the pixels which are closest to the
            # quantization level with the index of ind
        else:
            continue

        # print('minimum==ind\n',minimum==ind)
        # print('img[minimum==ind]\n',img[minimum==ind])
        # print('np.mean(img[minimum==ind],axis=0)\n',np.mean(img[minimum==ind],axis=0))

    # ========================
    return new_levels

def max_lloyd_quantize(img: np.ndarray, level_num: int, threshold: float, max_iter: int) -> Tuple[
    np.ndarray, np.ndarray]:
    '''

    Add your description and complete the inputs (params) and output (return).

    :param img: 3D numpy array with the shape of height X width X number of channels
    :param level_num: an int representing the number of the quantization levels
    :param threshold: an int representing a stop criteria/value for the algorithm
    :param max_iter: an int representing the maximal number of iterations
    :return new_levels: *********** the new quantization levels
    :return metric_vals: a numpy array representing the metric value in each iteration. In this function the metric values
                         is the maximal change between two corresponding f_is
    '''
    # ====== YOUR CODE: ======
    numOfChannels = img.shape[2]
    # the next line shows the original initiation method:
    # f = np.linspace(np.ones((1,numOfChannels))*np.amin(img),np.ones((1,numOfChannels))*np.amax(img),level_num) # initial guess of f
    # this is the new method, as we were asked to do in 2.4:
    f = init_levels(img, level_num)
    new_levels = f

    i = 0
    metric_vals = np.zeros(0)
    f_diff = 10000  # an arbitrary value
    print(i < max_iter and f_diff >= threshold)
    print(f_diff)
    print(threshold)
    while (i < max_iter and f_diff >= threshold):
        new_levels = max_lloyd_iter(img, new_levels)
        print("f-new_levels", f)
        f_diff = np.amax(f - new_levels)
        i = i + 1
        metric_vals = np.concatenate((metric_vals, np.array([f_diff])))
        print("dog")
        print(i < max_iter)
        print(f_diff)
        print(threshold)

    # ========================

    return new_levels, metric_vals


def quantize(img: np.ndarray, qunt_levels: np.ndarray, ) -> np.ndarray:
    '''
    Add your description and complete the inputs (params) and output (return).

    :param img: a 2D or 3D numpy array representing an image
    :param qunt_levels: a 2D numpy array representing quantization levels
    :return qunt_img: a 2D or 3D numpy array representing the original image after its pixels were quantized according to qunt_levels
    '''
    # ====== YOUR CODE: ======
    numOfChannels = qunt_levels.shape[1]
    level_num = qunt_levels.shape[0]
    qunt_levels = np.reshape(qunt_levels, (level_num, 1, 1, numOfChannels))  # preparing it for the next line
    distances = np.linalg.norm(img - qunt_levels,
                               axis=3)  # getting the distances of all the quantization levels to all pixels.
    #  We get an array with the size of number of quantization levels X image height X image width
    indices_of_minima = np.argmin(distances,
                                  axis=0)  # an array with the shape of the image. Each element in the array contains the
    # index of the closest quantization level to the pixel in the same place

    for ind in range(0, level_num):
        # new_levels[ind]=np.mean(img[indices_of_minima==ind],axis=0, dtype='float32')
        img[indices_of_minima == ind] = qunt_levels[ind]

    qunt_img = img
    # ========================
    return qunt_img

def init_levels(img: np.ndarray, level_num: int, ) -> np.ndarray:
    '''
    Add your description and complete the inputs (params) and output (return).

    :param img:
    :param level_num:
    :return init_vals:

    '''
    # ====== YOUR CODE: ======
    shape = img.shape
    np.random.seed(82)
    flag = 0
    rand_rows = np.random.randint(0, shape[0], level_num)
    rand_cols = np.random.randint(0, shape[1], level_num)
    init_vals = img[rand_rows, rand_cols, :]

    while flag == 0:
        if (np.unique(init_vals, axis=1).shape[0] < level_num):
            rand_rows = np.random.randint(0, shape[0], level_num)
            rand_cols = np.random.randint(0, shape[1], level_num)
            init_vals = img[rand_rows, rand_cols]
        else:
            flag = 1

    # ========================
    return init_vals

def main():
    # a = build_table(['SC-subjects.xls', 'ST-subjects.xls'])
    # print(a)

    wheel = colourwheel()
    iterNum = 40
    levels4, metrics4 = max_lloyd_quantize(wheel, 4, threshold=0.1, max_iter=iterNum)
    im4 = quantize(wheel, levels4)
    levels8, metrics8 = max_lloyd_quantize(wheel, 4, threshold=0.1, max_iter=iterNum)
    im8 = quantize(wheel, levels8)
    levels16, metrics16 = max_lloyd_quantize(wheel, 4, threshold=0.1, max_iter=iterNum)
    im16 = quantize(wheel, levels16)
    levels32, metrics32 = max_lloyd_quantize(wheel, 4, threshold=0.1, max_iter=iterNum)
    im32 = quantize(wheel, levels32)

    fig2 = plt.figure(figsize=(15, 15))
    plt.subplot(231)
    plt.imshow(wheel)
    plt.title('original')
    plt.subplot(232)
    plt.imshow(im4)
    plt.title('4 levels')
    plt.subplot(233)
    plt.imshow(im8)
    plt.title('8 levels')
    plt.subplot(234)
    plt.imshow(im16)
    plt.title('16 levels')
    plt.subplot(235)
    plt.imshow(im32)
    plt.title('32 levels')
    plt.show()

    plt.figure()
    plt.scatter(np.linspace(0, metrics4.size - 1, metrics4.size), metrics4, color='k', alpha=0.5)
    plt.scatter(np.linspace(0, metrics8.size - 1, metrics8.size), metrics8, color='b', alpha=0.5)
    plt.scatter(np.linspace(0, metrics16.size - 1, metrics16.size), metrics16, color='y', alpha=0.5)
    plt.scatter(np.linspace(0, metrics32.size - 1, metrics32.size), metrics32, color='r', alpha=0.5)
    plt.axhline(y=0.1)
    plt.show()

main()