"""

"""


# Built-in
import os

# Libs
from tqdm import tqdm

# Pytorch
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

# Own modules
import utils
import mobilenet, resnet, raspnet

# settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
epochs = 80
decay_step = [60]
decay_rate = 0.1
verb_step = 25
save_epoch = 5
batch_size = 64
class_num = 6
model_type = 'large'
model_name = 'mobilenet'
save_dir = './model/base/{}'.format(model_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if model_type == 'large':
    resize_shape = (224, 224)
else:
    resize_shape = (32, 32)


def main():
    # get data
    x_train, y_train, x_valid, y_valid = utils.get_images(r'./data/UTKFace', resize_shape=resize_shape)

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
    train_reader = DataLoader(utils.UTKDataLoader(x_train, y_train, tsfm=transform_train), batch_size=batch_size,
                              num_workers=4, shuffle=True)
    valid_reader = DataLoader(utils.UTKDataLoader(x_valid, y_valid, tsfm=transform_valid), batch_size=batch_size,
                              num_workers=4, shuffle=False)

    # network
    if model_type == 'large':
        if model_name == 'resnet':
            net = resnet.resnet34(True)
            net.fc = nn.Linear(512, class_num)
        elif model_name == 'mobilenet':
            net = mobilenet.mobilenet_v2(True)
            net.classifier[1] = nn.Linear(1280, class_num)
        else:
            raise NotImplementedError
    else:
        net = raspnet.raspnet(name=model_name, class_num=class_num)
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
                pbar.set_description('Epoch {} Step {}: train cross entropy loss: {:.4f}'.
                                     format(epoch + 1, i + 1, running_loss / verb_step))
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
        p, r, f1 = utils.f1_score(truth, pred, 0)
        print('Epoch {}: valid accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(
            epoch + 1, 100 * correct / total, p, r, f1))

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
