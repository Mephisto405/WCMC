import numpy as np
import imageio
import math
import os
import cv2
import matplotlib.pyplot as plt
import sys
import json
import csv
import torch

def StandardizeShape(images):

    origin_shape = images.shape

    if len(origin_shape) == 4:
        return images

    if len(origin_shape) == 2:
        return np.expand_dims(np.expand_dims(images, axis=0), axis=-1)

    if len(origin_shape) == 3:
        if origin_shape[2] <= 3:
            return np.expand_dims(images, axis=0)
        else:
            return np.expand_dims(images, axis=-1)

    assert(False)

def ToLDR(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = image.clip(0.0, 1.0)
        image = image * 255
        image = image.astype(np.uint8)
    return image

def ImgConvertTorchAndNumpy(tensor):


    if tensor is None:
        return None

    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy().copy()
        n_dim = len(tensor.shape)
        if n_dim >= 3:
            t = list(range(0, n_dim))
            t[-3:] = n_dim - 2, n_dim - 1, n_dim - 3
            tensor = np.transpose(tensor, t)
    else:
        n_dim = len(tensor.shape)
        if n_dim >= 3:
            t = list(range(0, n_dim))
            t[-3:] = n_dim - 1, n_dim - 3, n_dim - 2
            tensor = np.transpose(tensor, t)
        tensor = torch.from_numpy(tensor.copy())

    return tensor



def ImShow(images, title = 'title', wait=1, normalize=True):


    if isinstance(images, torch.Tensor):
        input = ImgConvertTorchAndNumpy(images)
    else:
        input = images.copy()

    input = StandardizeShape(input)


    if normalize == True:

        input = np.split(input, input.shape[0], axis=0)
        for i in range(len(input)):
            a = input[i]
            # a = a / np.max(a)
            if a.shape[-2] < 100:
                sp = (a.shape[-3] * 10, a.shape[-2] * 10)
                a = np.squeeze(a)
                a = cv2.resize(a, sp, interpolation = cv2.INTER_NEAREST)
            a = ToLDR(a)
            input[i] = a

    else:
        input = ToLDR(input)
        input = np.split(input, input.shape[0], axis=0)

    for i in range(len(input)):
        img = np.squeeze(input[i])
        cv2.imshow(title + '-' + str(i), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    cv2.waitKey(int(wait * 1000))

def MatShow(images, title = '', wait=1):

    input = StandardizeShape(images)
    input = np.split(input, input.shape[0], axis=0)

    for i in range(len(input)):
        img = np.squeeze(input[i])
        #plt.title(title + str(i))
        plt.figure(title + '-' + str(i))
        plt.imshow(img)
        plt.colorbar()

    plt.pause(wait * 1000)

def ReadImg(file):
    if (file.find('.pfm') != -1):
        image = imageio.imread(file, 'PFM-FI')
        return np.flipud(image)

    image = imageio.imread(file)
    return np.array(image)

def ToHDR(image):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image = image / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32)
        image = image / 65535.0
    return image


def MakeDir(file_path):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def WriteImg(file, images):

    inputs = StandardizeShape(images)
    if inputs.dtype == np.float64:
        inputs = inputs.astype(np.float32)

    MakeDir(file)
    if inputs.shape[0] == 1 and inputs.shape[-1] == 3:
        WriteSingleImg(file, inputs[0,:,:,:], '')
    else:
        b,h,w,c = inputs.shape
        inputs = np.split(inputs, indices_or_sections=b, axis=0)
        if b == 1:
            WriteSingleImg(file, inputs[0].reshape((h,w,c)), '')
        else:
            for i in len(inputs):
                WriteSingleImg(file, inputs[i].reshape((h,w,c)), '-b'+str(i))


def WriteSingleImg(file, image, prefix_b):

    isPfm = (file.find('.pfm') != -1)
    h, w, c = image.shape

    if not c == 3:
        num_images = math.ceil(int(c) / 3)
        file_pre, file_post = os.path.splitext(file)
        for i in range(num_images):
            ca = int(i * 3)
            cb = int(min((i + 1) * 3, c))
            part_image = np.zeros([h, w, 3], dtype=np.float32)
            part_image[:, :, 0: (cb - ca)] = image[:, :, ca: cb]

            if c == 1:
                pc = ''
            else:
                pc = '-p' + str(i)

            if (isPfm):
                imageio.imwrite(file_pre + prefix_b + pc + file_post, np.flipud(part_image), 'PFM-FI')
            else:
                imageio.imwrite(file_pre + prefix_b + pc + file_post, part_image)
    else:
        file_pre, file_post = os.path.splitext(file)

        if (isPfm):
            imageio.imwrite(file_pre + prefix_b + file_post, np.flipud(image), 'PFM-FI')
        else:
            imageio.imwrite(file_pre + prefix_b + file_post, image)


def WriteCSV(file, dict):

    with open(file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value, type(value)])


def ReadCSV(file):

    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        r_dict = dict(reader)

    print(r_dict)



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def WriteJson(file, dict):

    with open(file, 'w') as json_file:
        json.dump(dict, json_file, cls=MyEncoder)

def ReadJson(file):

    if not os.path.exists(file):
        print('Can not locate file : ' + file)
        sys.exit(1)

    with open(file) as json_file:
        dict = json.load(json_file)

    return dict

def PlotHistogram(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)


