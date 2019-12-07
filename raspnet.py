"""

"""


# Built-in

# Libs
from torch import nn

# Own modules


class AgeNet(nn.Module):
    def __init__(self, class_num):
        super(AgeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.3)

        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.3)

        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)

        self.fc = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.cls = nn.Linear(256, class_num)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view((-1, 64*4*4))
        x = self.relu(self.fc(x))
        x = self.drop(x)
        return self.cls(x)


class AgeNetM(nn.Module):
    def __init__(self, class_num):
        super(AgeNetM, self).__init__()
        self.conv1_d = nn.Conv2d(3, 3, 3, 1, 1, groups=3)
        self.conv1_p = nn.Conv2d(3, 32, 1, 1, 0)
        # self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.3)

        # self.conv21_d = nn.Conv2d(32, 32, 3, 1, 1, groups=32)
        # self.conv21_p = nn.Conv2d(32, 64, 1, 1, 0)
        self.conv21 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2_1 = nn.ReLU()
        # self.conv22_d = nn.Conv2d(64, 64, 3, 1, 1, groups=64)
        # self.conv22_p = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.3)

        # self.conv31_d = nn.Conv2d(64, 64, 3, 1, 1, groups=64)
        # self.conv31_p = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv31 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_1 = nn.ReLU()
        # self.conv32_d = nn.Conv2d(64, 64, 3, 1, 1, groups=64)
        # self.conv32_p = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)

        self.fc = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.cls = nn.Linear(256, class_num)

    def forward(self, x):
        x = self.relu1(self.conv1_p(self.conv1_d(x)))
        # x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # x = self.relu2_1(self.conv21_p(self.conv21_d(x)))
        # x = self.relu2_2(self.conv22_p(self.conv22_d(x)))
        x = self.relu2_1(self.conv21(x))
        x = self.relu2_2(self.conv22(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # x = self.relu3_1(self.conv31_p(self.conv31_d(x)))
        # x = self.relu3_2(self.conv32_p(self.conv32_d(x)))
        x = self.relu3_1(self.conv31(x))
        x = self.relu3_2(self.conv32(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view((-1, 64*4*4))
        x = self.relu(self.fc(x))
        x = self.drop(x)
        return self.cls(x)


def raspnet(**kwargs):
    if kwargs['name'] == 'agenet':
        model = AgeNet(kwargs['class_num'])
    # add your implementation here
    elif kwargs['name'] == 'agenetm':
        model = AgeNetM(kwargs['class_num'])
    else:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    pass
