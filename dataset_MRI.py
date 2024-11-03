import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib
import random
import numpy as np
import matplotlib as plt
import cv2

class Dataset_master():
    def __init__(self, args, ratio):
        self.file_name_list = os.listdir(args['dirpath'])

        tag = []
        for i in range(len(self.file_name_list)):
            tmp = self.file_name_list[i]
            tmp2 = tmp.split("_")
            if tmp2[0] != "BraTS2021":
                # not the file we are looking for
                tag.append(0)
            else:
                tag.append(1)
        self.file_name_list = [self.file_name_list[i] for i in range(len(self.file_name_list)) if tag[i] == 1]
        # remove non-data file

        index = []
        for i in range(len(self.file_name_list)):
            tmp = self.file_name_list[i]
            tmp2 = tmp.split("_")
            if tmp2[0] != "BraTS2021":
                # not the file we are looking for
                index.append(-1)
            else:
                index.append(int(tmp2[1]))

        sorted_indices = sorted(range(len(index)), key=lambda x: index[x])
        self.file_name_list = [self.file_name_list[i] for i in sorted_indices]

        indices = list(range(len(self.file_name_list)))

        sample_size = int(ratio * len(indices))
        selected_indices = random.sample(indices, sample_size)

        remaining_indices = list(set(indices) - set(selected_indices))

        train_list = [self.file_name_list[i] for i in selected_indices]
        test_list = [self.file_name_list[i] for i in remaining_indices]

        self.train = myDataset(args, train_list)
        self.test = myDataset(args, test_list)


class myDataset(Dataset):
    def __init__(self, args, file_list):
        self.args = args
        self.file_name_list = file_list

    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        data_file = os.path.join(self.args['dirpath'],file_name,file_name+'_t1.nii.gz') # test
        label_file = os.path.join(self.args['dirpath'],file_name,file_name+'_seg.nii.gz')

        data = nib.load(data_file).get_fdata()
        label = nib.load(label_file).get_fdata()

        data = torch.from_numpy(data)
        data = torch.nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), size=self.args['norm_size'], mode='trilinear',
                                                align_corners=False)
        data = data.squeeze(0)
        data  = (data - data.min()) / (data.max() - data.min() + 1e-5)
        data = torch.as_tensor(data, dtype=torch.float32)
        # data = data/(torch.max(data))
        # print("fffff")
        # print(torch.max(data))
        # print(torch.min(data))
        label = torch.from_numpy(label)
        label[label == 1] = 1
        label[label == 2] = 1
        label[label == 4] = 1

        label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0), size=self.args['norm_size'],
                                               mode='nearest')
        label = label.squeeze(0) #.squeeze(0)
        label = torch.as_tensor(label, dtype=torch.float32)

        # print(torch.unique(label))
        return data, label

    def __len__(self):
        return len(self.file_name_list)


if __name__ == "__main__":
    master = Dataset_master({'dirpath': '..\\data', 'norm_size': [80,80,52]}, 0.8)
    train_ds, test_ds = master.train, master.test
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=False)
    for i, (ct, label) in enumerate(train_dl):
        print(i, ct.dtype, ct.shape, label.shape)

    '''
    a = os.listdir("..\\data")
    file_name_list = os.listdir("..\\data")
    index = []
    for i in range(len(file_name_list)):
        tmp = file_name_list[i]
        tmp2 = tmp.split("_")
        if tmp2[0] != "BraTS2021":
            # not the file we are looking for
            index.append(-1)
        else:
            index.append(int(tmp2[1]))

    sorted_indices = sorted(range(len(index)), key=lambda x: index[x])
    file_name_list = [file_name_list[i] for i in sorted_indices]

    indices = list(range(len(file_name_list)))

    # 随机抽取80%的索引
    sample_size = int(0.8 * len(indices))
    selected_indices = random.sample(indices, sample_size)

    # 获取剩余的索引
    remaining_indices = list(set(indices) - set(selected_indices))

    # 根据索引分割列表
    train_list_tmp = [file_name_list[i] for i in selected_indices]
    test_list_tmp = [file_name_list[i] for i in remaining_indices]

    tag = []
    for i in range(len(train_list_tmp)):
        tmp = train_list_tmp[i]
        tmp2 = tmp.split("_")
        if tmp2[0] != "BraTS2021":
            # not the file we are looking for
            tag.append(0)
        else:
            tag.append(1)
    train_list = [train_list_tmp[i] for i in range(len(train_list_tmp)) if tag[i] == 1]

    tag = []
    for i in range(len(test_list_tmp)):
        tmp = test_list_tmp[i]
        tmp2 = tmp.split("_")
        if tmp2[0] != "BraTS2021":
            # not the file we are looking for
            tag.append(0)
        else:
            tag.append(1)
    test_list = [test_list_tmp[i] for i in range(len(test_list_tmp)) if tag[i] == 1]
    print(len(train_list))
    print(len(test_list)
          )
    print(train_list)
    print(test_list)

    file_name = 'BraTS2021_01408'
    data_file = os.path.join("..\\data", file_name, file_name + '_t2.nii.gz')  # test
    label_file = os.path.join("..\\data", file_name, file_name + '_seg.nii.gz')

    data = nib.load(data_file).get_fdata()
    label = nib.load(label_file).get_fdata()

    data1 = torch.from_numpy(data)
    data1 = torch.nn.functional.interpolate(data1.unsqueeze(0).unsqueeze(0), size=[120,120,75], mode='trilinear', align_corners=False)
    # data1 = torch.unsqueeze(data1,dim=0)
    print(data1.shape)
    '''
