from Code.model import *
from sklearn.externals import joblib
from Code.load_ucsd import *
from PIL import Image
from Code.util import *
from torch.autograd import Variable as variable
import cv2
import numpy as np
from sklearn import svm
from torch.autograd import variable as Variable
import torch.utils.data.distributed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.Mse = torch.nn.MSELoss(reduce=False, size_average=False)
        if self.args.index_model == 1:
            self.model_flow = VAE()
        elif self.args.index_model == 2:
            # if self.args.index_dataset == 2:
            #     self.model_flow = CAE_FLOW_Avenue()
            # else:
            #     self.model_flow = CAE_FLOW()
            self.model_flow = autoencoder()
            self.Mse = torch.nn.MSELoss(reduce=False, size_average=False)
        else:
            self.model_flow = CAE_RGB()  #ucsd2 used the model, but the memeory has increased very large
            self.Mse = torch.nn.MSELoss(reduce=False, size_average=False)
        self.pixel_label = []
        self.frame_label = []
        self.classifier = dict()
        self.train_latent = dict()
        self.test_latent = torch.zeros(0, self.args.hid_dim)  # hid_dim
        if self.args.index_dataset == 0:
            w = 118
            h = 78     #Ped1 model_one
        elif self.args.index_dataset == 1:
            w = 119
            h = 179  #Ped2 model_one
        else:
            w = 89
            h = 159    #Avenue model_one
        if self.args.index_model == 1:
            w = 15
            h = 22

        for i in range(w * h): # ped2 model_two
            self.classifier[i] = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
            self.train_latent[i] = torch.Tensor(0, self.args.hid_dim) # hide_dim

    def Reparameter(self, mu, logvar, Flag):
        if Flag == 1:
            parameter = Variable(torch.randn(mu.size())).cuda()
        else:
            parameter = Variable(torch.randn(mu.size()))
        return parameter * torch.exp(logvar/2) + mu
    def set_learning(self, optim):
        lr = 1e-3 * (0.1 ** (self.args.epoch // 50))
        for param_group in optim.param_groups:
            param_group['lr'] = lr
    def cae_flow_train(self, dataloader):
        optim = torch.optim.Adam(self.model_flow.parameters(), lr=self.args.lr)
        print('training......')
        L = 0
        for _ in range(self.args.epoch):
            L = L + 1
            print('epoch number:{}'.format(L))
            for img in dataloader:
                optim.zero_grad()
                if torch.cuda.is_available():
                    x = variable(img.cuda())
                else:
                    x = img
                #CAE net
                z, _ = self.model_flow.encoder(x)
                x_ = self.model_flow.decoder(z)
                loss = self.CAE_loss_function(x_, x)
                loss.backward()
                optim.step()
            print("loss_{}".format(loss.item()))
            # self.save_images(x_, x.shape[-2], x.shape[-1], x.shape[-3],
            #                  '../img/CAE/img_{}.png'.format(L), L)
            if L % 5 == 0:
                torch.save(self.model_flow.state_dict(), '../checkpoint/CAE/cae_flow_train_{}_{}_{}.pk'.format(
                    self.args.index_dataset, self.args.index_model, L))
        print("Training Done")
    def cae_rgb_train(self, dataloader):
        optim = torch.optim.Adam(self.model_flow.parameters(), lr=self.args.lr)
        print('training......')
        L = 0
        for _ in range(self.args.epoch):
            L = L + 1
            print('learning rate:{}'.format(optim.param_groups[0]['lr']))
            for img in dataloader:
                optim.zero_grad()
                if torch.cuda.is_available():
                    x = variable(img.cuda())
                else:
                    x = img
                #CAE net
                z = self.model_flow.encoder(x)
                x_ = self.model_flow.decoder(z)
                loss = self.CAE_loss_function(x_, x)
                loss.backward()
                optim.step()
            print("loss_{}".format(loss.item()))
            # save_images(input, h, w, c, path, index)
            self.save_images(x_, x.shape[-2], x.shape[-1], x.shape[-3],
                             './img/CAE/img_{}.png'.format(L), L)
            if L % 5 == 0:
                torch.save(self.model_flow.state_dict(), './checkpoint/CAE/cae_rgb_train_{}_{}_{}.pk'.format(
                    self.args.index_dataset, self.args.index_model, L))
        print("Training Done")
    def vae_train(self, dataloader):
        optim = torch.optim.Adam(self.model_flow.parameters(), lr=self.args.lr)
        print('training vae model......')
        L = 0
        for _ in range(self.args.epoch):
            L = L + 1
            print('learning rate:{}'.format(optim.param_groups[0]['lr']))
            for img in dataloader:
                optim.zero_grad()
                if torch.cuda.is_available():
                    x = variable(img.cuda())
                else:
                    x = img
                #VAE Net
                mu, logvar = self.model_flow.encoder(x)
                z = self.Reparameter(mu, logvar, 1)
                x_ = self.model_flow.decoder(z)
                loss = self.VAE_loss_function(x, x_, mu, logvar)
                loss.backward()
                optim.step()
            print("loss_{}".format(loss.item()))
            # save_images(input, h, w, c, path, index)
            self.save_images(x_, x.shape[-2], x.shape[-1], x.shape[-3],
                             './img/VAE/img_{}.png'.format(L), L)
            if L % 5 == 0:
                torch.save(self.model_flow.state_dict(), './checkpoint/VAE/vae_train_{}_{}_{}.pk'.format(
                    self.args.index_dataset, self.args.index_model, L))
        print("Training Done")

    # def OneClassSvm(self, dataloader):
    #     if self.args.pattern == 0:
    #         OCSVM_path = 'cae_flow_train'
    #     else:
    #         OCSVM_path = 'cae_rgb_train'
    #     self.model_flow.load_state_dict(torch.load('../checkpoint/CAE/' + OCSVM_path + '_{}_{}_{}.pk'.format(
    #         self.args.index_dataset, self.args.index_model, self.args.index_flow)))
    #     L = 0
    #     with torch.no_grad():
    #         for data in dataloader:
    #             # CAE net
    #             _, z_rgb = self.model_flow.encoder(data)
    #             for index_x in range(z_rgb.shape[-2]):
    #                 for index_y in range(z_rgb.shape[-1]):
    #                     index = z_rgb.shape[-1] * index_x + index_y
    #                     # z_total = torch.cat( (z_rgb[:, :, index_x, index_y], z_flow[:, :, index_x, index_y]), 1)
    #                     self.train_latent[index] = torch.cat((self.train_latent[index], z_rgb[:, :, index_x, index_y]), 0)
    #             L = L + 1
    #             print(len(dataloader))
    #             print("We have load the read {} rgb dataset ".format(L))
    #     for class_index in range(z_rgb.shape[-2] * z_rgb.shape[-1]):
    #         self.classifier[class_index].fit(self.train_latent[class_index])
    #     joblib.dump(self.classifier, "../OC_SVM_File/OneClassSvm_{}_{}.m".format(self.args.index_dataset, self.args.index_model))
    #     print("OneClassSvm train Done and Parameter Load Sucessful")

    def OneClassSvm(self, dataloader):
        if self.args.pattern == 0:
            OCSVM_path = 'cae_flow_train'
        else:
            OCSVM_path = 'cae_rgb_train'
        self.model_flow.load_state_dict(torch.load('../checkpoint/CAE/' + OCSVM_path + '_{}_{}_{}.pk'.format(
            self.args.index_dataset, self.args.index_model, self.args.index_flow)))
        L = 0
        with open('../Txt/data.txt', 'w') as f:
            with torch.no_grad():
                for data in dataloader:
                    # CAE net
                    _, z_rgb = self.model_flow.encoder(data)
                    f.write(str(z_rgb.numpy())+'\n')
                    # for index_x in range(z_rgb.shape[-2]):
                    #     for index_y in range(z_rgb.shape[-1]):
                    #         index = z_rgb.shape[-1] * index_x + index_y
                    #         # z_total = torch.cat( (z_rgb[:, :, index_x, index_y], z_flow[:, :, index_x, index_y]), 1)
                    #         self.train_latent[index] = torch.cat((self.train_latent[index], z_rgb[:, :, index_x, index_y]), 0)
                    L = L + 1
                    if L == 2:
                        break
                    print("We have load the read {} rgb dataset ".format(L))
        temp_tenosr = []
        with open('../Txt/data.txt', 'r') as f:
            for index_x in range(z_rgb.shape[-2]):
                for index_y in range(z_rgb.shape[-1]):
                    temp_tenosr.clear()
                    for num, line in enumerate(f):
                            print("num", num)
                            print('index',z_rgb.shape[-1] * index_x + index_y)
                            print(line[0])
                            print(line[:, :, index_x, index_y])
                            temp_tenosr.append(line[:, :, index_x, index_y])
                    temp_tenosr = torch.cat(temp_tenosr)
                    self.classifier[num].fit(temp_tenosr)
        for class_index in range(z_rgb.shape[-2] * z_rgb.shape[-1]):
            self.classifier[class_index].fit(self.train_latent[class_index])
        joblib.dump(self.classifier, "../OC_SVM_File/OneClassSvm_{}_{}.m".format(self.args.index_dataset, self.args.index_model))
        print("OneClassSvm train Done and Parameter Load Sucessful")


    def OneClassSvm_test(self, labels):
        gr_label = []
        pre_score = []
        temp_score = []
        total_score = []
        total_gr = []
        normal_point = []
        abnormal_point = []
        # pixel_list_2 = np.array([3,4,14,18,19,21,22,23,24,32])
        # pixel_list_2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
        #                          22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        pixel_list_2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        self.classifier = joblib.load("../OC_SVM_File/OneClassSvm_{}_{}.m".format(
            self.args.index_dataset, self.args.index_model))
        if self.args.pattern == 0:
            OCSVM_path = 'cae_flow_train'
            label_path = 'pre_label'
        else:
            OCSVM_path = 'cae_rgb_train'
            label_path='rgb_pre_label'
        self.model_flow.load_state_dict(torch.load('../checkpoint/CAE/' + OCSVM_path + '_{}_{}_{}.pk'.format(
            self.args.index_dataset, self.args.index_model, self.args.index_flow)))
        with torch.no_grad():
            for k in range(len(pixel_list_2)):
                test_flow, gr_image = Load_test_flow(pixel_list_2[k], self.args.index_dataset)
                pre_score.clear()
                gr_label.clear()
                for i in range(len(test_flow)):
                    x_flow = test_flow[i]
                    _, z_flow = self.model_flow.encoder(x_flow)
                    # img = np.zeros((z_flow.shape[-2], z_flow.shape[-1]))
                    temp_score.clear()
                    Sum  = 0
                    for index_x in range(z_flow.shape[-2]):
                        for index_y in range(z_flow.shape[-1]):
                            index = z_flow.shape[-1] * index_x + index_y
                            self.test_latent = z_flow.cpu()[:, :, index_x, index_y]
                            # pre_label = self.classifier[index].predict(self.test_latent)
                            Sum = Sum + (self.classifier[index].score_samples(self.test_latent))
                            # temp_score.append(self.classifier[index].score_samples(self.test_latent))
                            # if index_x == 100 and index_y == 100 and pre_label == -1:
                            #     abnormal_point.append(self.test_latent)
                            # if index_x == 100 and index_y == 100 and pre_label == 1:
                            #     normal_point.append(self.test_latent)
                            # if pre_label == -1:
                            #     img[index_x, index_y] = 1
                    pre_score.append(Sum / (z_flow.shape[-1] * z_flow.shape[-2]))
                    # pre_score.append(np.max(temp_score)) #we use the min data
                    gr_label.append(labels[pixel_list_2[k] - 1][i])
                    # img = resize_image(img)
                    # Image.fromarray((255 * img).astype('uint8')).save(
                    #             './'+ label_path + '/' + str(pixel_list_2[k]) + '/pre_label_{}.png'.format(i))
                    # if len(abnormal_point)>=1:
                    #     self.plt_3d_point(normal_point, abnormal_point)
                    print("we have complete the {} image".format(i))

                roc(gr_label, self.Normalizatio(pre_score), 0)
                # if k == 0:
                #     total_score = pre_score
                #     total_gr = gr_label
                # else:
                #     total_score = total_score + pre_score
                #     total_gr = total_gr + gr_label
                total_score = total_score + pre_score
                total_gr = total_gr + gr_label
                roc(total_gr, self.Normalizatio(total_score), 0)
        print("save the ROC figure and OneClassSvm test Done")


    def Normalizatio(self, x):
        min_num = np.min(x)
        max_num = np.max(x)
        for i in range(len(x)):
            x[i] = (x[i] - min_num)/(max_num - min_num)
        return x
    def VAE_loss_function(self,input, output, mu, logvar):
        MSE= torch.mean(torch.sum((input - output) ** 2,
                                    dim=tuple(range(1, input.dim()))))
        # MSE = F.binary_cross_entropy(input, output.detach(), size_average=False)
        KL_loss = 0.5 * torch.sum(-logvar + mu.pow(2) + logvar.exp() - 1)
        return MSE + KL_loss
    def CAE_loss_function(self, input, output):
        loss = torch.mean(torch.sum((input - output) ** 2,
                                    dim=tuple(range(1, input.dim()))))
        return loss
    def save_images(self, input, h, w, c, path, index):
        # save_images(input, h, w, c, path, index)
        x_split = input.split(1, 0)
        x_numpy = x_split[0].cpu().detach().numpy()
        x_numpy = x_numpy.swapaxes(1, 2).swapaxes(2, 3)
        x_numpy = x_numpy.reshape(h, w, c)
        split = cv2.split(x_numpy)
        Image.fromarray((255 * split[0]).astype('uint8')).save(
            path.format(index))
    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def plt_3d_point(self, normal_data, abnormal_data):
        normal_data, abnormal_data = torch.cat(normal_data).numpy(),torch.cat(abnormal_data).numpy()
        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        # 将数据点分成三部分画，在颜色上有区分度
        for i in range(len(normal_data)):
            ax.scatter(normal_data[i][0], normal_data[i][1], normal_data[i][2],s=5, c='b')  # 绘制数据点
        for j in range(len(abnormal_data)):
            ax.scatter(abnormal_data[j][0], abnormal_data[j][1], abnormal_data[j][2], s=8, c='r')  # 绘制数据点
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.savefig("../point_{}.png".format(0))
        # if index == 1:
        plt.show()
