"""

"""


# Built-in
import time

# Libs
import cv2
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN

# Pytorch
import torch
import torchvision.transforms as transforms

# Own modules
import raspnet


# age thresholds
age_thresh = (0, 6, 18, 25, 35, 60, 100)
age_str = ['{}~{}'.format(a, b) for (a, b) in zip(age_thresh[:-1], age_thresh[1:])]

# running device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initialize module
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
    device=device
)
# define network
model_name = 'agenet'
class_num = 6
model_dir = './model/base/small/{}/epoch-75.pth.tar'.format(model_name)
net = raspnet.raspnet(name=model_name, class_num=class_num)
net.load_state_dict(torch.load(model_dir)['state_dict'])
net.to(device)

tsfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# read video
# cap = cv2.VideoCapture('/Users/BohaoHuang/Google Drive/Others/embark/2019Fall/facenet-pytorch/examples/video.mp4')
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
ret = True

while ret:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret and frame is None:
        continue
    frame = cv2.resize(frame, (600, 360))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # detect boxes
    boxes, _ = mtcnn.detect(frame)
    draw = ImageDraw.Draw(frame)
    np_frame = np.array(frame)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)

            w0, h0, w1, h1 = box.tolist()
            w0, h0, w1, h1 = int(w0), int(h0), int(w1), int(h1)
            crop = np_frame[h0:h1, w0:w1, :]
            crop = tsfm(crop)
            outputs = net(torch.unsqueeze(crop, 0))
            _, predicted = torch.max(outputs.data, 1)
            pred = predicted.cpu().numpy()
            np_frame = np.array(frame)
            cv2.putText(np_frame, age_str[predicted], (w0, h0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    duration = time.time() - start_time

    frame = np_frame[:, :, ::-1]
    print('fps: {:.1f}'.format(1/duration))
    cv2.imshow('result', frame)
    cv2.waitKey(1)

cap.release()
