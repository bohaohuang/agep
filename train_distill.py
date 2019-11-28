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
lambda_ = 0.2
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
    # TODO

    # define reader
    # TODO

    # train
    for epoch in range(epochs):
        running_loss = 0.0
        cls = 0.0
        dst = 0.0
        pbar = tqdm(train_reader)
        for i, data in enumerate(pbar):
            # TODO train the file with distillation loss

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
                truth.extend(labels.cpu().numpy())
                pred.extend(predicted.cpu().numpy())

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
