import torch
from torch import nn
# 导入记好了，         2维卷积，2维最大池化，展成1维，全连接层，构建网络结构辅助工具,2d网络归一化,激活函数,自适应平均池化
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torchsummary import summary


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model0 = Sequential(
            # 0
            # 输入3通道、输出48通道、卷积核大小、步长、补零、
            Conv2d(in_channels=3, out_channels=48, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(48),
            ReLU(),
            Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(48),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(48),
            ReLU(),
            Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(48),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(96),
            ReLU(),
            Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(96),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(96),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(96),
            ReLU(),
            Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(96),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(192),
            ReLU(),
            Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(192),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=96, out_channels=192, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(192),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(192),
            ReLU(),
            Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(192),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(384),
            ReLU(),
            Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=192, out_channels=384, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(384),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
            Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(384),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(384, num_classes)

    def forward(self, x):
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def resnet1818(pretrained=False, pretrained_model_path=None, **kwargs):
    model = Resnet18(50)
    return model