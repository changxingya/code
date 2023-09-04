import torch.nn as nn
import torch.nn.functional as F
import torch
hid_dim = 480
min_num = 2
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # convolution layers
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)  # 73
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0, stride=1)  # 152
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0, stride=1)  # 146
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0, stride=1) # 70
        self.conv4 = nn.Conv2d(128, 256, 3, padding=0, stride=1) # 31
        self.conv5 = nn.Conv2d(256, 512, 3, padding=0, stride=1) #15
        self.reduce_dim_1 = nn.Conv2d(512, 256, 1, padding=0, stride=1)
        self.reduce_dim_2 = nn.Conv2d(256, 128, 1, padding=0, stride=1)
        self.mu = nn.Conv2d(128, 5, 1, padding=0, stride=1)
        self.logvar = nn.Conv2d(128, 5, 1, padding=0, stride=1)
        # deconvolution layers
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.increase_dim_1 = nn.Conv2d(5, 128, 1, padding=0, stride=1)
        self.increase_dim_2 = nn.Conv2d(128, 256, 1, padding=0, stride=1)
        self.increase_dim_3 = nn.Conv2d(256, 512, 1, padding=0, stride=1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=0)

    def encoder(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        self.size1 = x.size()
        x, self.indeces1 = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        self.size2 = x.size()
        x, self.indeces2 = self.pool(x)
        x = F.leaky_relu(self.conv4(x))
        self.size3 = x.size()
        x, self.indeces3 = self.pool(x)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.reduce_dim_1(x))
        x = F.leaky_relu(self.reduce_dim_2(x))
        mu = F.leaky_relu(self.mu(x))
        logvar = F.leaky_relu(self.logvar(x))
        return mu, logvar

    def decoder(self, x):
        x = F.leaky_relu(self.increase_dim_1(x))
        x = F.leaky_relu(self.increase_dim_2(x))
        x = F.leaky_relu(self.increase_dim_3(x))
        x = F.leaky_relu(self.deconv1(x))
        x = self.unpool(x, self.indeces3, self.size3)
        x = F.leaky_relu(self.deconv2(x))
        x = self.unpool(x, self.indeces2, self.size2)
        x = F.leaky_relu(self.deconv3(x))
        x = self.unpool(x, self.indeces1, self.size1)
        x = F.leaky_relu(self.deconv4(x))
        x = F.leaky_relu(self.deconv5(x))
        x = F.sigmoid(x)
        return x

class CAE_RGB(nn.Module):
    def __init__(self):
        super(CAE_RGB, self).__init__()
        #convolution
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)  # 73
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # 152
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # 146
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 146
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 146
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)  # 146
        self.reduce_dim_1 = nn.Conv2d(512, 256, 1, padding=0, stride=1)
        self.reduce_dim_2 = nn.Conv2d(256, 128, 1, padding=0, stride=1)
        self.mu = nn.Conv2d(128, 5, 1, padding=0, stride=1)
        # deconvolution layers
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.increase_dim_1 = nn.Conv2d(5, 128, 1, padding=0, stride=1)
        self.increase_dim_2 = nn.Conv2d(128, 256, 1, padding=0, stride=1)
        self.increase_dim_3 = nn.Conv2d(256, 512, 1, padding=0, stride=1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding= 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        self.size1 = x.size()
        x, self.indeces1 = self.pool(x)
        x = F.relu(self.conv3(x))
        self.size2 = x.size()
        x, self.indeces2 = self.pool(x)
        x = F.relu(self.conv4(x))
        self.size3 = x.size()
        x, self.indeces3 = self.pool(x)
        x = F.relu(self.conv5(x))
        self.size4 = x.size()
        x, self.indeces4 = self.pool(x)
        x = F.relu(self.reduce_dim_1(x))
        x = F.relu(self.reduce_dim_2(x))
        z = F.relu(self.mu(x))
        return z

    def decoder(self, x):
        x = F.relu(self.increase_dim_1(x))
        x = F.relu(self.increase_dim_2(x))
        x = F.relu(self.increase_dim_3(x))
        x = self.unpool(x, self.indeces4, self.size4)
        x = F.relu(self.deconv1(x))
        x = self.unpool(x, self.indeces3, self.size3)
        x = F.relu(self.deconv2(x))
        x = self.unpool(x, self.indeces2, self.size2)
        x = F.relu(self.deconv3(x))
        x = self.unpool(x, self.indeces1, self.size1)
        x = F.relu(self.deconv4(x))
        x = F.sigmoid(self.deconv5(x))
        return x

class CAE_FLOW(nn.Module):
    def __init__(self):
        super(CAE_FLOW, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(2, 32, 5, stride=1, padding=1)  # 152
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # 146
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1) # 70
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)  # 73
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=1) # 70
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, stride=1) # 70
        self.reduce_dim_1 = nn.Conv2d(256, 128, 1, padding=0, stride=1)
        self.reduce_dim_2 = nn.Conv2d(128, 64, 1, padding=0, stride=1)
        self.mu = nn.Conv2d(64, hid_dim, 1, padding=0, stride=1)
        # deconvolution layers
        self.increase_dim_1 = nn.Conv2d(hid_dim, 64, 1, padding=0, stride=1)
        self.increase_dim_2 = nn.Conv2d(64, 128, 1, padding=0, stride=1)
        self.increase_dim_3 = nn.Conv2d(128, 256, 1, padding=0, stride=1)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 2, kernel_size=5, stride=1, padding=1)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.size1 = x.size()
        x1, self.indeces1 = self.pool(x)
        x2 = F.relu(self.conv4(x1))
        x3 = F.relu(self.conv5(x2))
        x4 = F.relu(self.reduce_dim_1(x3))
        x5 = F.relu(self.reduce_dim_2(x4))
        x = F.relu(self.mu(x5))
        return x

    def decoder(self, x):
        x = F.relu(self.increase_dim_1(x))
        x = F.relu(self.increase_dim_2(x))
        x = F.relu(self.increase_dim_3(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.unpool(x, self.indeces1, self.size1)
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.sigmoid(self.deconv5(x))
        return x

class CAE_FLOW_Avenue(nn.Module):
    def __init__(self):
        super(CAE_FLOW_Avenue, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(2, 32, 5, stride=1, padding=1)  # 152
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # 146
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1) # 70
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)  # 73
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=1) # 70
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, stride=1) # 70
        self.reduce_dim_1 = nn.Conv2d(256, 128, 1, padding=0, stride=1)
        self.reduce_dim_2 = nn.Conv2d(128, 64, 1, padding=0, stride=1)
        self.mu = nn.Conv2d(64, hid_dim, 1, padding=0, stride=1)
        # deconvolution layers
        self.increase_dim_1 = nn.Conv2d(hid_dim, 64, 1, padding=0, stride=1)
        self.increase_dim_2 = nn.Conv2d(64, 128, 1, padding=0, stride=1)
        self.increase_dim_3 = nn.Conv2d(128, 256, 1, padding=0, stride=1)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 2, kernel_size=5, stride=1, padding=1)

    def encoder(self, x):

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.size1 = x.size()
        x, self.indeces1 = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        self.size2 = x.size()
        x, self.indeces2 = self.pool(x)
        x = F.relu(self.reduce_dim_1(x))
        x = F.relu(self.reduce_dim_2(x))
        x = F.relu(self.mu(x))
        return x

    def decoder(self, x):
        x = F.relu(self.increase_dim_1(x))
        x = F.relu(self.increase_dim_2(x))
        x = F.relu(self.increase_dim_3(x))
        x = self.unpool(x, self.indeces2, self.size2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.unpool(x, self.indeces1, self.size1)
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.sigmoid(self.deconv5(x))
        return x


class autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2,return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.reduce_pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(min_num, 16, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)

        # Decoder
        self.deconv0 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, padding=1, stride=1)
        self.deconv3 = nn.ConvTranspose2d(16, min_num, 3, padding=1, stride=1)

    def encoder(self, x):
        x = F.elu(self.conv1(x))
        self.size1 = x.size()
        x, self.indeces1 = self.pool(x)
        reduce_x1 = self.reduce_pool(self.reduce_pool(self.reduce_pool(x)))


        x = F.elu(self.conv2(x))
        self.size2 = x.size()
        x, self.indeces2 = self.pool(x)
        reduce_x2 = self.reduce_pool(self.reduce_pool(x))

        x = F.elu(self.conv3(x))
        self.size3 = x.size()
        x, self.indeces3 = self.pool(x)

        re_x = torch.cat((x, reduce_x2), 1)
        re_x = torch.cat((re_x, reduce_x1), 1)

        return x, re_x
    def decoder(self, x):

        x = F.relu(self.deconv0(x))
        x = self.unpool(x, self.indeces3, self.size3)
        x = F.relu(self.deconv1(x))
        x = self.unpool(x, self.indeces2, self.size2)
        x = F.relu(self.deconv2(x))
        x = self.unpool(x, self.indeces1, self.size1)
        x = F.sigmoid(self.deconv3(x))

        return x
    def forward(self, x):
        z, cat_x = self.encoder(x)
        print(cat_x.shape)
        x_hat = self.decoder(z)
        return x_hat
