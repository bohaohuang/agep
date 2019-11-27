"""

"""


# Built-in
import os
from glob import glob

# Libs
import numpy as np
from tqdm import tqdm
from skimage import io, color, transform

# Pytorch
import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader

# Own modules
import mobilenet

# settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
epochs = 80
decay_step = [60]
decay_rate = 0.1
verb_step = 25
save_epoch = 5
save_dir = './model/ref'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class UTKDataLoader(data.Dataset):
    def __init__(self, img, lbl, tsfm=None):
        self.img = img
        self.lbl = lbl
        self.transforms = tsfm

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.img[index]
        lbl = self.lbl[index]
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(lbl)


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


def get_images(parent_path, age_thresh=(6, 18, 25, 35, 60), valid_percent=0.2):
    img_files = sorted(glob(os.path.join(parent_path, '*.chip.jpg')))
    imgs, ages = [], []
    age_thresh = [-1, *age_thresh, 200]
    for img_file in tqdm(img_files):
        img = io.imread(img_file)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        age = int(os.path.splitext(os.path.basename(img_file))[0].split('_')[0])
        img = transform.resize(img, (224, 224), anti_aliasing=True, preserve_range=True)[..., :3]
        imgs.append(img.astype(np.uint8))

        for cnt, (lb, ub) in enumerate(zip(age_thresh[:-1], age_thresh[1:])):
            if lb < age <= ub:
                ages.append(cnt)
                break
    rand_idx = np.random.permutation(np.arange(len(img_files)))
    imgs = [imgs[a] for a in rand_idx]
    ages = [ages[a] for a in rand_idx]
    valid_num = int(np.floor(len(imgs) * valid_percent))
    return imgs[valid_num:], ages[valid_num:], imgs[:valid_num], ages[:valid_num]


def main():
    # get data
    x_train, y_train, x_valid, y_valid = get_images(r'./data/UTKFace')
    # define reader
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_reader = DataLoader(UTKDataLoader(x_train, y_train, tsfm=transform_train), num_workers=4, shuffle=True)
    valid_reader = DataLoader(UTKDataLoader(x_valid, y_valid, tsfm=transform_valid), num_workers=4, shuffle=True)

    # network
    # net = AgeNet(6)
    # net.to(device)
    net = mobilenet.mobilenet_v2(True)
    net.classifier[1] = nn.Linear(1280, 6)
    net.to(device)

    # define loss
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    # train
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_reader)
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % verb_step == verb_step - 1:
                pbar.set_description('Epoch {} Step {}: train cross entropy loss: {:.4f}'.format(epoch + 1, i + 1,
                                                                                                 running_loss / verb_step))
                running_loss = 0.0

        # validation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_reader:
                inputs, labels = data
                inputs, labels = inputs.float().to(device), labels.long().to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Epoch {}: valid accuracy: {:.2f}'.format(epoch + 1, 100 * correct / total))

        if epoch % save_epoch == 0 and epoch != 0:
            save_name = os.path.join(save_dir, 'epoch-{}.pth.tar'.format(epoch))
            torch.save({
                'epoch': epochs,
                'state_dict': net.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, save_name)
            print('Saved model at {}'.format(save_name))

    print('Finished training')


if __name__ == '__main__':
    main()
