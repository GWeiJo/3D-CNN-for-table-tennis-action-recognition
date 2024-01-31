from torch import nn
import torch
from pytorch_i3d import InceptionI3d
import x3d as resnet_x3d
I3D_weight_RGB_pth = "I3D_weights/rgb_imagenet.pt" # path of I3D model weight
X3D_RGB_weight_pth = "X3D_models/x3d_multigrid_kinetics_fb_pretrained.pt" # path of X3D model weight
BS = 8
BS_UPSCALE = 1 # CHANGE WITH GPU AVAILABILITY
GPUS = 1
BASE_BS_PER_GPU = BS * BS_UPSCALE // GPUS # FOR SPLIT BN
CONST_BN_SIZE = 8
X3D_VERSION = 'M' # ['S', 'M', 'XL']

class X3D_transfer(nn.Module): # X3D transfer learning
    def __init__(self, fre, num_classes):
        super(X3D_transfer, self).__init__()
        X3D = self.transfer_learning(fre)
        self.X3D = X3D
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        output = self.X3D(x)
        output = self.flatten(output)
        # print(output.shape)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

    def transfer_learning(self, f_layer):
        X3D = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3,
                                    dropout=0.5, base_bn_splits=BASE_BS_PER_GPU//CONST_BN_SIZE)
        # X3D.load_state_dict(torch.load(X3D_RGB_weight_pth))
        counter = 0
        fre = 0
        if f_layer == 0:
            print("Freeze {} layer of model.".format(counter))
            return X3D
        if f_layer != 0:
            for child in X3D.children():
                counter = counter + 1
                if counter <= f_layer:
                    fre = fre + 1
                    for para in child.parameters():
                        para.requires_grad = False
                #         print(para.requires_grad)
                # if counter >= f_layer:
                #     for para in child.parameters():
                #         print(para.requires_grad)
            print("Import model has {} layers.".format(counter))
            print("Freeze {} layer of model.".format(fre))
            print("-----------------------------------------------")
            return X3D

class X3D_train_from_scratch(nn.Module): # X3D training from scratch
    def __init__(self, num_classes):
        super(X3D_train_from_scratch, self).__init__()
        X3D = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=num_classes, n_input_channels=3,
                                    dropout=0.5, base_bn_splits=BASE_BS_PER_GPU//CONST_BN_SIZE)
        self.X3D = X3D
    def forward(self, x):
        output = self.X3D(x)
        print(output)
        return output

class I3D_transfer_2(nn.Module): # I3D transfer learning 
    def __init__(self, fre, num_classes):
        super(I3D_transfer_2, self).__init__()
        I3D = self.transfer_learning(fre)
        self.I3D = I3D
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(11200, 1024) #11200 in 60fps, 8000 in 45fps, 6400 in 36fps
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        output = self.I3D(x)
        output = self.flatten(output)
        # print(output.shape)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

    def transfer_learning(self, f_layer):
        I3D = InceptionI3d()
        I3D.load_state_dict(torch.load(I3D_weight_RGB_pth)) # load I3D model weight
        counter = 0
        fre = 0
        if f_layer == 0:
            print("Freeze {} layer of model.".format(counter))
            return I3D
        if f_layer != 0:
            for child in I3D.children():
                counter = counter + 1
                if counter <= f_layer:
                    fre = fre + 1
                    for para in child.parameters():
                        para.requires_grad = False
                #         print(para.requires_grad)
                # if counter >= f_layer:
                #     for para in child.parameters():
                #         print(para.requires_grad)
            print("Import model has {} layers.".format(counter))
            print("Freeze {} layer of model.".format(fre))
            print("-----------------------------------------------")
            return I3D

class I3D_transfer(nn.Module): # I3D transfer learning 
    def __init__(self, fre, num_classes):
        super(I3D_transfer, self).__init__()
        I3D = self.transfer_learning(fre)
        self.I3D = I3D
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(11200, num_classes) #11200 in 60fps, 8000 in 45fps, 6400 in 36fps

    def forward(self, x):
        output = self.I3D(x)
        output = self.flatten(output)
        output = self.fc1(output)
        return output

    def transfer_learning(self, f_layer):
        I3D = InceptionI3d()
        I3D.load_state_dict(torch.load(I3D_weight_RGB_pth)) # load I3D model weight
        counter = 0
        fre = 0
        if f_layer == 0:
            print("Freeze {} layer of model.".format(counter))
            return I3D
        if f_layer != 0:
            for child in I3D.children():
                counter = counter + 1
                if counter <= f_layer:
                    fre = fre + 1
                    for para in child.parameters():
                        para.requires_grad = False
                #         print(para.requires_grad)
                # if counter >= f_layer:
                #     for para in child.parameters():
                #         print(para.requires_grad)
            print("Model has {} layers.".format(counter))
            print("Freeze {} layer of model.".format(fre))
            print("-----------------------------------------------")
            return I3D

# arch series is the model I builded based on vgg16

class arch23(nn.Module):
    def __init__(self, num_classes):
        super(arch23, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(128, 128)
        self.Conv_layer6 = self.convlayer_3d_3(128, 128)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer7 = self.convlayer_3d_3(128, 128)
        self.Conv_layer8 = self.convlayer_3d_3(128, 128)
        self.maxpooling4 = nn.MaxPool3d((2, 1, 1))
        # self.Conv_layer9 = self.convlayer_3d_3(256, 256)
        # self.Conv_layer10 = self.convlayer_3d_3(256, 256)
        # self.maxpooling5 = nn.MaxPool3d((2, 1, 1))
        self.flatten = nn.Flatten()
        self.layer_fc1   = nn.Linear(98304, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer5(x)
        x = self.Conv_layer6(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer7(x)
        x = self.Conv_layer8(x)
        x = self.maxpooling4(x)
        # x = self.Conv_layer9(x)
        # x = self.Conv_layer10(x)
        # x = self.maxpooling5(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch22_pretrain_UCF101(nn.Module):
    def __init__(self, num_classes):
        super(arch22_pretrain_UCF101, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer13 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.Conv_layer12 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        self.maxpooling5 = nn.MaxPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(50176, num_classes)

    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer13(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.Conv_layer12(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        x = self.maxpooling5(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch22_avgpool(nn.Module):
    def __init__(self, num_classes):
        super(arch22_avgpool, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.AvgPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.AvgPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer13 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.AvgPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.Conv_layer12 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.AvgPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        self.maxpooling5 = nn.AvgPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1   = nn.Linear(50176, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer13(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.Conv_layer12(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        x = self.maxpooling5(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch22(nn.Module):
    def __init__(self, num_classes):
        super(arch22, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer13 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.Conv_layer12 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        self.maxpooling5 = nn.MaxPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1   = nn.Linear(50176, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer13(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.Conv_layer12(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        x = self.maxpooling5(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch21(nn.Module):
    def __init__(self, num_classes):
        super(arch21, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer13 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.Conv_layer12 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        self.maxpooling5 = nn.MaxPool3d((1, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1   = nn.Linear(125440, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer13(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.Conv_layer12(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        x = self.maxpooling5(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch20(nn.Module):
    def __init__(self, num_classes):
        super(arch20, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer13 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.Conv_layer12 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        self.flatten = nn.Flatten()
        self.layer_fc1   = nn.Linear(50176, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer13(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.Conv_layer12(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch19(nn.Module):
    def __init__(self, num_classes):
        super(arch19, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 128)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer2 = self.convlayer_3d_3(128, 64)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 64)
        self.maxpooling3 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer4 = self.convlayer_3d_3(64, 64)
        self.maxpooling4 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(64, 64)
        self.Conv_layer6 = self.convlayer_2d(64, 64)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(138240, num_classes)

    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer3(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer5(x)       
        x = self.Conv_layer6(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d
    
    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch18(nn.Module):
    def __init__(self, num_classes):
        super(arch18, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        # self.maxpooling5 = nn.MaxPool3d((2, 1, 1))
        self.flatten = nn.Flatten()
        # self.layer_fc1 = nn.Linear(, 1024)
        self.layer_fc2 = nn.Linear(163840, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        # x = self.maxpooling5(x)
        print(x.shape)
        x = self.flatten(x)
        # x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch17(nn.Module):
    def __init__(self, num_classes):
        super(arch17, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.Conv_layer11 = self.convlayer_2d(512, 512)
        self.maxpooling5 = nn.MaxPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(12800, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer10(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer11(x)
        x = self.maxpooling5(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch16(nn.Module):
    def __init__(self, num_classes):
        super(arch16, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_2d(256, 512)
        self.Conv_layer9 = self.convlayer_2d(512, 512)
        self.Conv_layer_2d_1 = self.convlayer_2d(512, 512)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(12800, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        # print(x.shape)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.Conv_layer10(x)
        x = self.maxpooling2(x)
        # print(x.shape)
        x = self.Conv_layer5(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.maxpooling3(x)
        # print(x.shape)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.Conv_layer_2d_1(x)
        x = self.maxpooling4(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch15(nn.Module):
    def __init__(self, num_classes):
        super(arch15, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer_2d_1 = self.convlayer_2d(512, 512)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(4608, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        # print(x.shape)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.Conv_layer10(x)
        x = self.maxpooling2(x)
        # print(x.shape)
        x = self.Conv_layer5(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.maxpooling3(x)
        # print(x.shape)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.maxpooling4(x)
        x = self.Conv_layer_2d_1(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch14(nn.Module):
    def __init__(self, num_classes):
        super(arch14, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.Conv_layer10 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(12800, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        # print(x.shape)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.Conv_layer10(x)
        x = self.maxpooling2(x)
        # print(x.shape)
        x = self.Conv_layer5(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.maxpooling3(x)
        # print(x.shape)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.maxpooling4(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch13(nn.Module):
    def __init__(self, num_classes):
        super(arch13, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.Conv_layer7 = self.convlayer_3d_3(256, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer8 = self.convlayer_3d_3(256, 512)
        self.Conv_layer9 = self.convlayer_3d_3(512, 512)
        self.maxpooling4 = nn.MaxPool3d((2, 1, 1))
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(123904, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        # print(x.shape)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        # print(x.shape)
        x = self.Conv_layer5(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)
        x = self.maxpooling3(x)
        # print(x.shape)
        x = self.Conv_layer8(x)
        x = self.Conv_layer9(x)
        x = self.maxpooling4(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch12(nn.Module):
    def __init__(self, num_classes):
        super(arch12, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 128)
        self.Conv_layer4 = self.convlayer_3d_3(128, 128)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(128, 256)
        self.Conv_layer6 = self.convlayer_3d_3(256, 256)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer7 = self.convlayer_3d_3(256, 512)
        self.Conv_layer8 = self.convlayer_3d_3(512, 512)
        self.maxpooling4 = nn.MaxPool3d((2, 2, 2))
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(36864, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        # print(x.shape)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        # print(x.shape)
        x = self.Conv_layer5(x)
        x = self.Conv_layer6(x)
        x = self.maxpooling3(x)
        # print(x.shape)
        x = self.Conv_layer7(x)
        x = self.Conv_layer8(x)
        x = self.maxpooling4(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch11(nn.Module):
    def __init__(self, num_classes):
        super(arch11, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 64)
        self.Conv_layer2 = self.convlayer_3d_3(64, 64)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 96)
        self.Conv_layer4 = self.convlayer_3d_3(96, 96)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(96, 96)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(96, 128)
        self.Conv_layer7 = self.convlayer_3d_3(128, 128)
        self.Conv_layer8 = self.convlayer_2d(128, 128)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(301056, 1024)
        self.layer_fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)       
        x = self.Conv_layer8(x)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch10(nn.Module):
    def __init__(self, num_classes):
        super(arch10, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 128)
        self.Conv_layer2 = self.convlayer_3d_3(128, 64)
        self.maxpooling1 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 64)
        self.Conv_layer4 = self.convlayer_3d_3(64, 64)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(64, 64)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer6 = self.convlayer_3d_3(64, 64)
        self.Conv_layer7 = self.convlayer_3d_3(64, 64)
        self.Conv_layer8 = self.convlayer_2d(64, 64)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(302848, num_classes)

    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer3(x)
        x = self.Conv_layer4(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer5(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer6(x)
        x = self.Conv_layer7(x)       
        x = self.Conv_layer8(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_5(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 5, 5), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_7(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 7, 7), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch9(nn.Module):
    def __init__(self, num_classes):
        super(arch9, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 128)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer2 = self.convlayer_3d_3(128, 64)
        self.maxpooling2 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 64)
        self.maxpooling3 = nn.MaxPool3d((2, 2, 2))
        self.Conv_layer4 = self.convlayer_3d_3(64, 64)
        # self.maxpooling4 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(64, 64)
        self.Conv_layer6 = self.convlayer_2d(64, 64)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(188160, num_classes)

    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer3(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer4(x)
        # x = self.maxpooling4(x)
        x = self.Conv_layer5(x)       
        x = self.Conv_layer6(x)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_5(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 5, 5), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_7(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 7, 7), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch7(nn.Module):
    def __init__(self, num_classes):
        super(arch7, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 128) # kernel_size = 7 x 7 x 3
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer2 = self.convlayer_3d_3(128, 64) # kernel_size = 7 x 6 x 3
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 64)
        self.maxpooling3 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer4 = self.convlayer_3d_3(64, 64)
        self.Conv_layer5 = self.convlayer_2d(64, 64)
        # self.Conv_layer6 = self.convlayer_2d(64, 64)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(147456, num_classes)
        # self.layer_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer3(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer4(x)
        x = self.Conv_layer5(x)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        # x = self.layer_softmax
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_5(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 5, 5), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_7(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 7, 7), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

class arch8(nn.Module):
    def __init__(self, num_classes):
        super(arch8, self).__init__()
        self.Conv_layer1 = self.convlayer_3d_3(3, 128)
        self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer2 = self.convlayer_3d_3(128, 64)
        self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer3 = self.convlayer_3d_3(64, 64)
        self.maxpooling3 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer4 = self.convlayer_3d_3(64, 64)
        # self.maxpooling4 = nn.MaxPool3d((1, 2, 2))
        self.Conv_layer5 = self.convlayer_3d_3(64, 64)
        self.Conv_layer6 = self.convlayer_2d(64, 64)
        self.flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(752640, num_classes)

    def forward(self, x):
        x = self.Conv_layer1(x)
        x = self.maxpooling1(x)
        x = self.Conv_layer2(x)
        x = self.maxpooling2(x)
        x = self.Conv_layer3(x)
        x = self.maxpooling3(x)
        x = self.Conv_layer4(x)
        # x = self.maxpooling4(x)
        x = self.Conv_layer5(x)       
        x = self.Conv_layer6(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.layer_fc1(x)
        return x
    
    def convlayer_3d_3(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_5(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 5, 5), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_3d_7(self, input, output):
        conv_layer_3d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(3, 7, 7), padding=1),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_3d

    def convlayer_2d(self, input, output):
        conv_layer_2d = nn.Sequential(
            nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
            # nn.ReLU(),
            nn.ReLU(),
            nn.BatchNorm3d(output),
        )
        return conv_layer_2d

# class arch6(nn.Module):
#     def __init__(self, num_classes):
#         super(arch6, self).__init__()
#         # self.Conv7_layer = self.convlayer_3d_7(3, 64) # kernel_size = 7 x 7 x 3
#         # self.Conv5_layer = self.convlayer_3d_5(3, 64)
#         self.Conv3_layer1 = self.convlayer_3d_3(3, 32)
#         self.maxpooling1 = nn.MaxPool3d((1, 2, 2))
#         self.Conv3_layer2 = self.convlayer_3d_3(32, 32)
#         self.maxpooling2 = nn.MaxPool3d((1, 2, 2))
#         self.Conv3_layer3 = self.convlayer_3d_3(32, 32)
#         self.Conv3_layer4 = self.convlayer_3d_3(32, 32)
#         self.maxpooling3 = nn.MaxPool3d((1, 2, 2))
#         self.Conv3_layer5 = self.convlayer_3d_3(32, 32)
#         # self.Conv_layer2 = self.convlayer_2d(64, 64)
#         self.flatten = nn.Flatten()
#         self.layer_fc1 = nn.Linear(401408, num_classes)
#         # self.layer_softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # x = self.Conv7_layer(x)
#         # x = self.maxpooling1(x)
#         # x = self.Conv5_layer(x)
#         x = self.Conv3_layer1(x)    
#         x = self.maxpooling1(x)   
#         x = self.Conv3_layer2(x)
#         x = self.maxpooling2(x)  
#         x = self.Conv3_layer3(x)
#         x = self.Conv3_layer4(x)
#         x = self.maxpooling3(x)
#         x = self.Conv3_layer5(x)
#         x = self.flatten(x)
#         x = self.layer_fc1(x)
#         # x = self.layer_softmax
#         return x
    
#     def convlayer_3d_3(self, input, output):
#         conv_layer_3d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             # nn.MaxPool3d((2, 2, 2), stride=1),
#         )
#         return conv_layer_3d

#     def convlayer_3d_5(self, input, output):
#         conv_layer_3d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(3, 5, 5), padding=1),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             # nn.MaxPool3d((2, 2, 2), stride=1),
#         )
#         return conv_layer_3d

#     def convlayer_3d_7(self, input, output):
#         conv_layer_3d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(3, 7, 7), padding=1),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             # nn.MaxPool3d((2, 2, 2), stride=1),
#         )
#         return conv_layer_3d

#     def convlayer_2d(self, input, output):
#         conv_layer_2d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             # nn.MaxPool3d((1, 2, 2), stride=1),
#         )
#         return conv_layer_2d


# class arch5(nn.Module):
#     def __init__(self, num_classes):
#         super(arch5, self).__init__()
#         self.Conv_layer1 = self.convlayer_3d_7(3, 128) # kernel_size = 7 x 7 x 3
#         self.Conv_layer2 = self.convlayer_3d_7(128, 64) # kernel_size = 7 x 6 x 3
#         self.Conv_layer3 = self.convlayer_3d_5(64, 64)
#         self.Conv_layer4 = self.convlayer_3d_3(64, 64)
#         self.Conv_layer5 = self.convlayer_2d(64, 64)
#         self.Conv_layer6 = self.convlayer_2d(64, 64)
#         self.flatten = nn.Flatten()
#         self.layer_fc1 = nn.Linear(1168128, num_classes)
#         # self.layer_softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.Conv_layer1(x)
#         x = self.Conv_layer2(x)
#         x = self.Conv_layer3(x)
#         x = self.Conv_layer4(x)
#         x = self.Conv_layer5(x)
#         x = self.flatten(x)
#         x = self.layer_fc1(x)
#         # x = self.layer_softmax
#         return x
    
#     def convlayer_3d_3(self, input, output):
#         conv_layer_3d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(3, 3, 3), padding=1),
#             # nn.ReLU(),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             nn.MaxPool3d((2, 2, 2), stride=1),
#         )
#         return conv_layer_3d

#     def convlayer_3d_5(self, input, output):
#         conv_layer_3d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(3, 5, 5), padding=1),
#             # nn.ReLU(),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             nn.MaxPool3d((2, 2, 2), stride=1),
#         )
#         return conv_layer_3d

#     def convlayer_3d_7(self, input, output):
#         conv_layer_3d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(3, 7, 7), padding=1),
#             # nn.ReLU(),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             nn.MaxPool3d((2, 2, 2), stride=1),
#         )
#         return conv_layer_3d

#     def convlayer_2d(self, input, output):
#         conv_layer_2d = nn.Sequential(
#             nn.Conv3d(input, output, kernel_size=(1, 3, 3)),
#             # nn.ReLU(),
#             nn.ReLU(),
#             nn.BatchNorm3d(output),
#             nn.MaxPool3d((1, 2, 2), stride=1),
#         )
#         return conv_layer_2d


# class arch1(nn.Module):
#     def __init__(self, num_classes):
#         super(arch1, self).__init__()
#         self.layer1 = nn.Conv3d(3, 64, kernel_size=3) # kernel_size = 3 x 3 x 3
#         self.batch64=nn.BatchNorm1d(64)
#         self.layer_ReLU = nn.ReLU()
#         self.layer2 = nn.Conv3d(64, 64, kernel_size=3) # kernel_size = 3 x 3 x 3
#         self.batch64=nn.BatchNorm1d(64)
#         self.layer_ReLU =  nn.ReLU()
#         self.layer3 = nn.Conv3d(64, 32, kernel_size=3) # kernel_size = 3 x 3 x 3
#         self.batch32=nn.BatchNorm1d(32)
#         self.layer_ReLU =  nn.ReLU()        
#         self.flatten = nn.Flatten()
#         self.layer_fc1 = nn.Linear(3041536, num_classes)
#         # self.layer_softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.batch64(x)
#         x = self.layer_ReLU(x)
#         x = self.layer2(x)
#         x = self.batch64(x)
#         x = self.layer_ReLU(x)
#         x = self.layer3(x)
#         x = self.batch32(x)
#         x = self.layer_ReLU(x)
#         x = self.flatten(x)
#         x = self.layer_fc1(x)
#         # x = self.layer_softmax(x)
#         return x

# class arch2(nn.Module):
#     def __init__(self, num_classes):
#         super(arch2, self).__init__()
#         self.Conv_layer1 = nn.Conv3d(3, 32, kernel_size=(3, 7, 7)) # kernel_size = 7 x 7 x 3
#         self.layer_ReLU = nn.ReLU()
#         self.MP_layer1 = nn.MaxPool3d((2, 2, 2), stride=1)
#         self.Conv_layer2 = nn.Conv3d(32, 64, kernel_size=(3, 7, 6)) # kernel_size = 7 x 6 x 3
#         self.layer_ReLU =  nn.ReLU()
#         self.MP_layer2 = nn.MaxPool3d((3, 3, 3), stride=1)
#         self.Conv_layer3 = nn.Conv3d(64, 32, kernel_size=(1, 7, 4))
#         self.layer_ReLU = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.layer_fc1 = nn.Linear(1344672, num_classes)
#         # self.layer_softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.Conv_layer1(x)
#         x = self.layer_ReLU(x)
#         x = self.MP_layer1(x)
#         x = self.Conv_layer2(x)
#         x = self.layer_ReLU(x)
#         x = self.MP_layer2(x)
#         x = self.Conv_layer3(x)
#         x = self.layer_ReLU(x)
#         x = self.flatten(x)
#         x = self.layer_fc1(x)
#         # x = self.layer_softmax
#         return x

# class arch3(nn.Module):
#     def __init__(self, num_classes):
#         super(arch3, self).__init__()
#         self.Conv_layer1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5)) # kernel_size = 7 x 7 x 3
#         self.layer_ReLU = nn.ReLU()
#         self.Conv_layer2 = nn.Conv3d(32, 64, kernel_size=(3, 7, 7)) # kernel_size = 7 x 6 x 3
#         self.layer_ReLU =  nn.ReLU()
#         self.MP_layer2 = nn.MaxPool3d((2, 2, 2), stride=1)
#         self.Conv_layer3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3))
#         self.layer_ReLU = nn.ReLU()
#         self.Conv_layer4 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3))
#         self.layer_ReLU = nn.ReLU()
#         self.Conv_layer5 = nn.Conv3d(64, 32, kernel_size=(1, 3, 3))
#         self.layer_ReLU = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.layer_fc1 = nn.Linear(1371168, 128)
#         self.layer_fc2 = nn.Linear(128, num_classes)
#         # self.layer_softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.Conv_layer1(x)
#         x = self.layer_ReLU(x)
#         # x = self.MP_layer1(x)
#         x = self.Conv_layer2(x)
#         x = self.layer_ReLU(x)
#         x = self.MP_layer2(x)
#         x = self.Conv_layer3(x)
#         x = self.layer_ReLU(x)
#         x = self.Conv_layer4(x)
#         x = self.layer_ReLU(x)
#         x = self.Conv_layer5(x)
#         x = self.layer_ReLU(x)
#         x = self.flatten(x)
#         x = self.layer_fc1(x)
#         x = self.layer_fc2(x)
#         # x = self.layer_softmax
#         return x

# class arch4(nn.Module):
#     def __init__(self, num_classes):
#         super(arch4, self).__init__()
#         self.conv_layer1 = self._make_conv_layer(3, 32)
#         self.conv_layer2 = self._make_conv_layer(32, 64)
#         # self.conv_layer3 = self._make_conv_layer(64, 124)
#         # self.conv_layer4 = self._make_conv_layer(124, 256)
#         # self.conv_layer5 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=0)
        
#         self.fc5 = nn.Linear(559872, 512)
#         self.ReLU = nn.ReLU()
#         self.batch0=nn.BatchNorm1d(512)
#         self.drop=nn.Dropout(p=0.15)        
#         self.fc6 = nn.Linear(512, 256)
#         self.ReLU = nn.ReLU()
#         self.batch1=nn.BatchNorm1d(256)
        
#         self.drop=nn.Dropout(p=0.15)
#         self.fc7 = nn.Linear(256, num_classes)

#     def _make_conv_layer(self, in_c, out_c):
#         conv_layer = nn.Sequential(
#         nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
#         nn.BatchNorm3d(out_c),
#         nn.ReLU(),
#         # nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
#         # nn.BatchNorm3d(out_c),
#         # nn.ReLU(),
#         nn.MaxPool3d((2, 2, 2)),
#         )
#         return conv_layer

#     def forward(self, x):
#         x = self.conv_layer1(x)
#         #print(x.size())
#         x = self.conv_layer2(x)
#         #print(x.size())
#         # x = self.conv_layer3(x)
#         #print(x.size())
#         # x = self.conv_layer4(x)
#         #print(x.size())
#         # x=self.conv_layer5(x)
#         #print(x.size())
#         x = x.view(x.size(0), -1)
#         x = self.fc5(x)
#         x = self.ReLU(x)
#         x = self.batch0(x)
#         x = self.drop(x)
#         x = self.fc6(x)
#         x = self.ReLU(x)
#         x = self.batch1(x)
#         x = self.drop(x)
#         # x1=x
#         x = self.fc7(x)
#         return x
#         # return x,x1

