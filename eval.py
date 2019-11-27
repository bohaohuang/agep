"""

"""


# Built-in

# Libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# Pytorch
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

# Own modules
import reader
import resnet, mobilenet

# Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = 'mobilenet'
model_dir = './model/{}/epoch-5.pth.tar'.format(network)
age_thresh = 6
batch_size = 500
data_dir = r'./data'


if __name__ == '__main__':
    # define network
    if network == 'resnet':
        net = resnet.resnet18(True)
        net.fc = nn.Linear(512, 2)
    elif network == 'mobilenet':
        net = mobilenet.mobilenet_v2(True)
        net.classifier[1] = nn.Linear(1280, 2)
    else:
        raise NotImplementedError
    net.load_state_dict(torch.load(model_dir)['state_dict'])
    net.to(device)

    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    valid_reader = DataLoader(reader.UTKDataLoader(data_dir, 'valid.hdf5', transforms=transform_valid,
                                                   age_thresh=age_thresh),
                              batch_size=batch_size, num_workers=4)

    truth = []
    pred = []
    conf = []
    with torch.no_grad():
        for data in tqdm(valid_reader):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            truth.extend(labels.cpu().numpy())
            pred.extend(predicted.cpu().numpy())
            conf.extend(outputs[:, 1].cpu().numpy())
    truth, pred, conf = np.array(truth), np.array(pred), np.array(conf)

    fpr, tpr, _ = roc_curve(truth, conf)
    auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
