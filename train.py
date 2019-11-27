"""

"""


# Built-in
import os

# Libs
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

# Pytorch
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Own modules
import reader
import resnet
import mobilenet

# settings
learning_rate = 1e-2
data_dir = r'./data'
age_thresh = 6
batch_size = 64
epochs = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
verb_step = 25
save_epoch = 5
network = 'resnet'
utk_face = True
# weights = [1, 20]
save_dir = './model/{}'.format(network)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def main():
    # define network
    if network == 'resnet':
        net = resnet.resnet34(True)
        for param in net.parameters():
            param.requires_grad = False
        net.fc = nn.Sequential(nn.Linear(512, 100),
                               nn.Linear(100, 2))
        # net.fc = nn.Linear(2048, 2)
    elif network == 'mobilenet':
        net = mobilenet.mobilenet_v2(True)
        net.classifier[1] = nn.Linear(1280, 2)
    else:
        raise NotImplementedError
    net.to(device)

    # define reader
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if not utk_face:
        train_loader = reader.UTKDataLoader(data_dir, 'train.hdf5', transforms=transform_train, age_thresh=age_thresh)
        valid_loader = reader.UTKDataLoader(data_dir, 'valid.hdf5', transforms=transform_valid, age_thresh=age_thresh)
    else:
        train_loader = ImageFolder(os.path.join(data_dir, 'UTKFace', 'train'), transform=transform_train)
        valid_loader = ImageFolder(os.path.join(data_dir, 'UTKFace', 'valid'), transform=transform_valid)
    train_reader = DataLoader(train_loader, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_reader = DataLoader(valid_loader, batch_size=batch_size, num_workers=4)

    # define loss
    weights = train_loader.class_cnt[1] / np.array(train_loader.class_cnt)
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
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
                pbar.set_description('Epoch {} Step {}: train cross entropy loss: {:.4f}'.format(epoch+1, i+1, running_loss/verb_step))
                running_loss = 0.0

        # validation
        correct = 0
        total = 0
        truth = []
        pred = []
        with torch.no_grad():
            for data in valid_reader:
                inputs, labels = data
                inputs, labels = inputs.float().to(device), labels.long().to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                truth.extend(labels.cpu().numpy())
                pred.extend(predicted.cpu().numpy())
        precision = precision_score(truth, pred)
        recall = recall_score(truth, pred)
        f1 = 2 * (precision * recall) / (precision + recall)
        print('Epoch {}: valid accuracy: {:.2f}'.format(epoch+1, 100*correct/total))
        print('\tPrecision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(precision, recall, f1))

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
