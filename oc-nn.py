import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score


data_fold = 'data'
if not os.path.isdir(data_fold):
    os.makedirs(data_fold)
train_set = MNIST(root=data_fold, train=True, download=True)
test_set = MNIST(root=data_fold, train=False, download=True)
train_data = train_set.train_data.numpy()
train_label = train_set.train_labels.numpy()
normal_train = train_data[np.where(train_label == 4), :, :]
normal_train = normal_train.transpose(1, 0, 2, 3)
normal_set = torch.FloatTensor(normal_train / 255.)
train_loader = DataLoader(normal_set, shuffle=True, batch_size=32, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_size = len(normal_train)
cae_model_path = './CAE.pth'
nu = 0.04
theta = 0.1
LR = 0.001
lr_list=[]
num_epochs = 100
class OC_NN(nn.Module):
    def __init__(self):
        super(OC_NN, self).__init__()
        self.dense_out1 = nn.Linear(1568, 32)
        self.out2 = nn.Linear(32, 1)
    def forward(self, img):
        img = img.view(img.shape[0], -1)
        w1 = F.relu(self.dense_out1(img))
        w2 = F.sigmoid(self.out2(w1))
        return w1, w2
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dense1 = nn.Linear(392, 32)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = x.view(-1, 392)
        x = self.dense1(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        return x
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dense1 = nn.Linear(32, 392)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
    def forward(self, encode):
        x = self.dense1(encode)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        x = x.view(x.size(0), 8, 7, 7)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.upsample2(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.upsample1(x)
        x = self.deconv1(x)
        x = F.sigmoid(x)
        return x
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.estimate = OC_NN()
    def relative_euclidean_distance(self, a, b):
        # return (a - b).norm(2, dim=1) / a.norm(2, dim=1)
        return (a - b).norm(2, dim=1)

    def forward(self, img):
        enc = self.encoder(img)
        dec = self.decoder(enc)
        rec_cosine = F.cosine_similarity(img, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(img, dec)
        z = torch.cat([rec_euclidean.unsqueeze(1), rec_cosine.unsqueeze(1)], dim=1)
        w1, w2 = self.estimate(z)
        return enc, dec, w1, w2, z
def nnscore(x, w, v):
    return torch.matmul(torch.matmul(x, w), v)
def ocnn_loss(x, nu, w1, w2, r):
    term1 = 0.5 * torch.sum(w1 ** 2)
    term2 = 0.5 * torch.sum(w2 ** 2)
    term3 = 1 / nu * torch.mean(F.relu(r - nnscore(x, w1, w2)))
    term4 = -r
    return term1 + term2 + term3 + term4
def save_image(inputs):
    x_split = inputs.split(1, 0)
    x_numpy = x_split[1].cpu().detach().numpy()
    x_numpy = x_numpy.swapaxes(1, 2).swapaxes(2, 3)
    x_numpy = x_numpy.reshape(inputs.shape[-2], inputs.shape[-1], 1)
    split = cv2.split(x_numpy)
    Image.fromarray((255 * split[0]).astype('uint8')).save(
        './img/img_{}.png'.format(epoch))
if __name__ == '__main__':
    model = CAE()
    model = model.to(device)
    encoder = model.encoder
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(num_epochs):
        count = 0
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs in train_loader:
            if (count ==  len(train_loader) // 32):
                break
            count = count + 1
            inputs = inputs.to(device)
            optimizer.zero_grad()
            enc, dec, w1, w2, z = model(inputs)
            encoder_tensor = encoder(inputs)
            mse = F.mse_loss(inputs, dec, size_average=False)
            print(w1)
            print(w2)
            r = nnscore(encoder_tensor, w1, w2)
            print("r", r)
            ocnn = ocnn_loss(encoder_tensor, nu, w1, w2, r)
            loss = theta * mse + (1-theta)*ocnn.mean()
            loss.backward()
            optimizer.step()
            # running_loss += loss.item() * inputs.size(0)
        # r = r.cpu().detach().numpy()
        if num_epochs % 10 == 0:
            save_image(dec)
        last_r = np.percentile(r.cpu().detach().numpy(), q=100 * nu)
        epoch_loss = loss
        print('Loss: {:.6f} '.format(epoch_loss))
        print('Mse: {:.6f}'.format(mse))
        print('OCNN: {:.6f}'.format(ocnn.mean()))

    train_set = MNIST(root=data_fold, train=True, download=True)
    test_set = MNIST(root=data_fold, train=False, download=True)

    test_data = test_set.train_data.numpy()
    test_label = test_set.train_labels.numpy()

    # normal class - 4
    class4 = test_data[np.where(test_label == 4), :, :]
    class4 = class4.transpose(1, 0, 2, 3)

    rand_idx = np.random.choice(len(class4), 220)
    class4 = class4[rand_idx, :, :, :]

    # anomaly class - 0, 7, 9
    class0 = test_data[np.where(test_label == 0), :, :]
    class0 = class0.transpose(1, 0, 2, 3)
    rand_idx = np.random.choice(len(class0), 5)
    class0 = class0[rand_idx, :, :, :]

    class7 = test_data[np.where(test_label == 7), :, :]
    class7 = class7.transpose(1, 0, 2, 3)
    rand_idx = np.random.choice(len(class7), 5)
    class7 = class7[rand_idx, :, :, :]

    normal_class = class4
    Class = np.concatenate((class4, class0, class7), axis=0)
    normal_encode = torch.FloatTensor(Class / 255.)

    y_train = np.ones(220)
    y_test = np.zeros(10)
    y_true = np.concatenate((y_train, y_test))

    y_score = []

    test_loader = DataLoader(normal_encode, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    for inputs in test_loader:
        inputs = inputs.to(device)
        enc, dec, w1, w2, z = model(inputs)
        # save_image(dec)
        encoder_tensor = encoder(inputs)
        mse = F.mse_loss(inputs, dec, size_average=False)
        r = nnscore(encoder_tensor, w1.squeeze(0), w2)
        ocnn = ocnn_loss(encoder_tensor, nu, w1.squeeze(0), w2, r)
        loss = theta * mse + (1 - theta) * ocnn.mean()
        score = loss.cpu().detach().numpy()
        y_score.append(r.mean().cpu().detach().numpy())
        print('score: {:.6f} '.format(r.mean().cpu().detach().numpy()))

    roc_score = roc_auc_score(y_true, y_score)

    print(roc_score)



