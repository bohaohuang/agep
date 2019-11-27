"""

"""


# Built-in
import os

# Libs
from tqdm import tqdm

# Pytorch
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Own modules
import resnet
import mobilenet
import raspnet
import utils

# settings
learning_rate = 1e-3
data_dir = r'./data'
age_thresh = 6
batch_size = 100
epochs = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
verb_step = 25
save_epoch = 1
save_dir = './model/dst'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
network_t = 'mobilenet'
network_s = 'agenet'
class_num = 6
teacher_dir = './model/base/large/epoch-5.pth.tar'
lambda_ = 1
temperature = 5


def make_network(model_name, class_num):
    if model_name == 'resnet':
        net = resnet.resnet34(True)
        net.fc = nn.Linear(512, class_num)
    elif model_name == 'mobilenet':
        net = mobilenet.mobilenet_v2(True)
        net.classifier[1] = nn.Linear(1280, class_num)
    else:
        net = raspnet.raspnet(name=model_name, class_num=class_num)
    return net


def main():
    # get data
    x_train, y_train, x_valid, y_valid = utils.get_images(r'./data/UTKFace', resize_shape=(224, 224))

    # define network
    net_t = make_network(network_t, class_num)
    net_t.load_state_dict(torch.load(teacher_dir)['state_dict'])
    for param in net_t.parameters():
        param.requires_grad = False
    net_t.to(device)

    net_s = make_network(network_s, class_num)
    net_s.to(device)

    # define loss
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net_t.parameters(), lr=learning_rate, momentum=0.9)

    # define reader
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_reader = DataLoader(utils.UTKDataLoaderDistill(x_train, y_train, 32, tsfm=transform_train),
                              batch_size=batch_size, num_workers=4, shuffle=True)
    valid_reader = DataLoader(utils.UTKDataLoaderDistill(x_valid, y_valid, 32, tsfm=transform_valid),
                              batch_size=batch_size, num_workers=4, shuffle=False)

    # train
    for epoch in range(epochs):
        running_loss = 0.0
        cls = 0.0
        dst = 0.0
        pbar = tqdm(train_reader)
        for i, data in enumerate(pbar):
            inputs_l, inputs_s, labels = data
            inputs_l, inputs_s, labels = inputs_l.float().to(device), inputs_s.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs_t = net_t(inputs_l)
            outputs_s = net_s(inputs_s)
            loss_cls = criterion(outputs_s, labels)
            loss_dst = F.kl_div(F.log_softmax(outputs_s / temperature), F.softmax(outputs_t / temperature),
                                reduction="batchmean")
            loss = loss_cls + lambda_ * loss_dst
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            cls += loss_cls.item()
            dst += loss_dst.item()
            if i % verb_step == verb_step - 1:
                pbar.set_description('Epoch {} Step {}: train cross entropy loss: {:.4f}, distilling loss: {:.4f}'.
                                     format(epoch+1, i+1, cls/verb_step, dst/verb_step))
                running_loss = 0.0

        # validation
        correct = 0
        total = 0
        truth = []
        pred = []
        with torch.no_grad():
            for data in valid_reader:
                inputs_l, inputs_s, labels = data
                inputs_l, inputs_s, labels = inputs_l.float().to(device), inputs_s.float().to(device), labels.long().to(device)
                outputs = net_s(inputs_s)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        p, r, f1 = utils.f1_score(truth, pred, 0)
        print('Epoch {}: valid accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(
            epoch + 1, 100 * correct / total, p, r, f1))

        if epoch % save_epoch == 0 and epoch != 0:
            save_name = os.path.join(save_dir, 'epoch-{}.pth.tar'.format(epoch))
            torch.save({
                'epoch': epochs,
                'state_dict': net_t.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, save_name)
            print('Saved model at {}'.format(save_name))

    print('Finished training')


if __name__ == '__main__':
    main()
