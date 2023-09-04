import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import scipy.io as sio
from PIL import  Image

def load_Ped1_label():
    labels = np.ones(shape=(36, 199))
    labels[0][60:152] = -1
    labels[1][50:175] = -1
    labels[2][91:199] = -1
    labels[3][31:168] = -1
    labels[4][list(range(5, 90)) + list(range(140, 199))] = -1
    labels[5][list(range(1, 100)) + list(range(110, 199))] = -1
    labels[6][0:175] = -1
    labels[7][0:94] = -1
    labels[8][0:48] = -1
    labels[9][0:140] = -1
    labels[10][70:165] = -1
    labels[11][130:199] = -1
    labels[12][0:156] = -1
    labels[13][0:199] = -1
    labels[14][138:199] = -1
    labels[15][123:199] = -1
    labels[16][0:47] = -1
    labels[17][54:120] = -1
    labels[18][64:138] = -1
    labels[19][45:175] = -1
    labels[20][31:199] = -1
    labels[21][16:107] = -1
    labels[22][8:165] = -1
    labels[23][50:171] = -1
    labels[24][40:135] = -1
    labels[25][77:144] = -1
    labels[26][10:122] = -1
    labels[27][105:199] = -1
    labels[28][list(range(1, 15)) + list(range(45, 113))] = -1
    labels[29][175:199] = -1
    labels[30][0:180] = -1
    labels[31][list(range(1, 52)) + list(range(65, 115))] = -1
    labels[32][5:165] = -1
    labels[33][0:121] = -1
    labels[34][86:199] = -1
    labels[35][15:108] = -1
    return labels

def load_Ped2_label():
    labels = np.ones(shape=(12, 179))
    labels[0][61:179] = -1
    labels[1][95:179] = -1
    labels[2][0:146] = -1  #149
    labels[3][31:179] = -1
    labels[4][0:129] = -1 #149
    labels[5][0:159] = -1
    labels[6][46:179] = -1
    labels[7][0:179] = -1
    labels[8][0:119] = -1 #119
    labels[9][0:149] = -1 #149
    labels[10][0:179] = -1
    labels[11][88:179] = -1
    return labels

def load_Avenue_label():
    labels = np.ones(shape=(21, 1437))
    labels[0][list(range(77, 119)) + list(range(391, 421)) + list(range(502, 665)) + list(range(867, 909)) + list(range(931, 1100))] = -1
    labels[1][list(range(272, 319)) + list(range(723, 763)) + list(range(1050, 1099))] = -1
    labels[2][list(range(294, 339)) + list(range(581, 621))] = -1
    labels[3][list(range(379, 427)) + list(range(648, 691))] = -1
    labels[4][list(range(468, 785))] = -1
    labels[5][list(range(344, 624)) + list(range(855, 1006))] = -1
    labels[6][list(range(422, 493)) + list(range(562, 594))] = -1
    labels[7][list(range(20, 29))] = -1
    labels[8][list(range(135, 182)) + list(range(495, 565)) + list(range(740, 754)) + list(range(874, 980))
              + list(range(1012, 1043)) + list(range(1103, 1162))] = -1
    labels[9][list(range(570, 606)) + list(range(636, 655)) + list(range(677, 712)) + list(range(723, 754))
              +list(range(782, 817))] = -1
    labels[10][list(range(20, 163)) + list(range(307, 345))] = -1
    labels[11][list(range(538, 616)) + list(range(644, 728)) + list(range(758, 842))] = -1
    labels[12][list(range(258, 285)) + list(range(457, 509))] = -1
    labels[13][list(range(398, 454)) + list(range(484, 499))] = -1
    labels[14][list(range(497, 586))] = -1
    labels[15][list(range(631, 729))] = -1
    labels[16][list(range(20, 55)) + list(range(98, 419))] = -1
    labels[17][list(range(20, 284))] = -1
    labels[18][list(range(108, 239))] = -1
    labels[19][list(range(64, 143)) + list(range(167, 240))] = -1
    labels[20][list(range(13, 65))] = -1
    return labels

def generate_avenue_pixel_lable():
    # load the ground truth mat file
    mat_data = sio.loadmat('./data/Avenue/testing_label_mask/1_label.mat')
    for i in range(len(mat_data['volLabel'][0])):
        Image.fromarray((255 * (mat_data['volLabel'][0][i])).astype('uint8')).save(
                './pre_label/' + str(3) + '/pre_label_{}.png'.format(i))

class Flow_Dataset(Dataset):
    def __init__(self, index_dataset, transform):
        super(Flow_Dataset, self).__init__()
        Img_file = []
        FileList = []
        Img_num = []
        root = '../data'
        if index_dataset == 0:
            txt_path = 'ped1_train'
            dir_path = '/UCSD_flow/UCSDped1/UCSDflow-train/'
        elif index_dataset == 1:
            txt_path = 'ped2_train'
            dir_path = '/UCSD_flow/UCSDped2/UCSD2flow-train/'
        elif index_dataset == 2:
            txt_path = 'Avenue_train'
            dir_path = '/Avenue/AvenueTrain/'
        for line in open('../Txt/'+ txt_path +'.txt', 'r'):
            Img_file.append(line.rstrip("\n"))
        for num in range(len(Img_file)):
            length = len(os.listdir(root + dir_path + 'flows/u/' + str(Img_file[num])))
            Img_num.append(length + 1)
        for i in range(len(Img_file)):
            j = 1
            while j < Img_num[i]:
                u_path = root + dir_path + 'flows/u/' + str(Img_file[i]) + '/flow_' + str(j).zfill(5) + '.jpg'
                FileList.append(u_path)
                v_path = root + dir_path + 'flows/v/' + str(Img_file[i]) + '/flow_' + str(j).zfill(5) + '.jpg'
                FileList.append(v_path)
                j = j + 1
        self.FileList = FileList
        self.transform = transform
    def __len__(self):
        return len(self.FileList)//2
    def __getitem__(self, index):
        u_data = cv2.imread(self.FileList[2 * index], 0)
        v_data = cv2.imread(self.FileList[2 * index + 1], 0)
        img = u_data
        img = cv2.merge([img, v_data])
        if self.transform is not None:
            img = self.transform(img)
        return img

def Load_test_flow(index, index_dataset):
    batch_data = []
    FileList = []
    Img_num = []
    gr_list = []
    if index_dataset == 0:
        txt_path = '../Txt/ped1_test.txt'
        file_path = '../data/UCSD_flow/UCSDped1/UCSDflow-test'
    elif index_dataset == 1:
        txt_path = '../Txt/ped2_test.txt'
        file_path = '../data/UCSD_flow/UCSDped2/UCSD2flow-test'
    else:
        txt_path = '../Txt/Avenue_test.txt'
        file_path = '../data/Avenue/AvenueTest'
    for line in open(txt_path, 'r'):
        FileList.append(line.rstrip("\n"))
    for num in range(len(FileList)):
        length = len(os.listdir(file_path + '/flows/u/' + str(FileList[num])))
        Img_num.append(length)
    # for i in range(len(FileList)):
    for i in range(1):
        print("Test_flow_{}".format(index))
        j = 1
        while j < (Img_num[index-1] + 1) :
            u_path = file_path + '/flows/u/'+str(FileList[index-1]) +'/flow_'+str(j).zfill(5)+'.jpg'
            v_path = file_path + '/flows/v/'+str(FileList[index-1]) +'/flow_'+str(j).zfill(5)+'.jpg'
            u_data = cv2.imread(u_path, 0)
            v_data = cv2.imread(v_path, 0)
            normal_data = (u_data / 255)
            v_data = (v_data / 255)
            normal_data = cv2.merge([normal_data, v_data])
            output = normal_data.swapaxes(1, 2).swapaxes(0, 1)
            tensor_output = torch.Tensor(output)
            tensor_temp = tensor_output.unsqueeze(0)
            batch_data.append(tensor_temp)
            j = j + 1
    print("the test data load done")
    return batch_data, gr_list

class RGB_Dataset(Dataset):
    def __init__(self, index_dataset, transform):
        super(RGB_Dataset, self).__init__()
        Img_file = []
        FileList = []
        Img_num = []
        channel = 1
        root = './data'
        if index_dataset == 0:
            txt_path = 'ped1_train'
            dir_path = '/UCSD_flow/UCSDped1/Train/'
        elif index_dataset == 1:
            txt_path = 'ped2_train'
            dir_path = '/UCSD_flow/UCSDped2/Train/'
        elif index_dataset == 2:
            txt_path = 'Avenue_train'
            dir_path = '/Avenue/Train/'
        for line in open('./Txt/'+ txt_path +'.txt', 'r'):
            Img_file.append(line.rstrip("\n"))
        for num in range(len(Img_file)):
            length = len(os.listdir(root + dir_path + str(Img_file[num])))
            Img_num.append(length + 1)
        for i in range(len(Img_file)):
            j = 1
            while j < Img_num[i]-2:
                for frame in range(channel):
                    path = root + dir_path + str(Img_file[i]) + '/' + str(j+frame).zfill(3) + '.tif'
                    FileList.append(path)
                j = j + 1
        self.FileList = FileList
        self.transform = transform
        self.channel = channel
    def __len__(self):
        return len(self.FileList)//self.channel
    def __getitem__(self, index):
        for i in range(self.channel):
            frame = cv2.imread(self.FileList[self.channel * index + i], 0)
            if i == 0 :
                img = frame
            else:
                img = cv2.merge([img, frame])
        if self.transform is not None:
            img = self.transform(img)
        return img

def Load_test_rgb(index, index_dataset):
    batch_data = []
    FileList = []
    Img_num = []
    gr_list = []
    if index_dataset == 0:
        txt_path = './Txt/ped1_test.txt'
        file_path = './data/UCSD_flow/UCSDped1/Test/'
    elif index_dataset == 1:
        txt_path = './Txt/ped2_test.txt'
        file_path = './data/UCSD_flow/UCSDped2/Test/'
    else:
        txt_path = './Txt/Avenue_test.txt'
        file_path = './data/Avenue/Test/'
    for line in open(txt_path, 'r'):
        FileList.append(line.rstrip("\n"))
    for num in range(len(FileList)):
        length = len(os.listdir(file_path + '/' + str(FileList[num])))
        Img_num.append(length)
    # for i in range(len(FileList)):
    for i in range(1):
        print("Test_flow_{}".format(index))
        j = 1
        while j < (Img_num[index-1] - 2) :
            path_one = file_path + str(FileList[index-1]) + '/' + str(j).zfill(3)+'.tif'
            # path_two = file_path + str(FileList[index-1]) + '/' + str(j+1).zfill(3)+'.tif'
            # path_three= file_path + str(FileList[index-1]) + '/' + str(j+2).zfill(3)+'.tif'

            frame_one = ( cv2.imread(path_one, 0) )/255
            # frame_two = ( cv2.imread(path_two, 0) ) /255
            # frame_three = ( cv2.imread(path_three, 0) ) /255

            normal_data = np.expand_dims(frame_one, 0)


            # normal_data = cv2.merge([frame_one, frame_two])
            # normal_data = cv2.merge([normal_data, frame_three])


            tensor_output = torch.Tensor(normal_data)
            tensor_temp = tensor_output.unsqueeze(0)
            batch_data.append(tensor_temp)
            j = j + 1
    print("the test data load done")
    return batch_data, gr_list
