import torch
import kornia
import numpy as np
import pandas as pd
import cv2
import clip
from PIL import Image
from datetime import datetime
from torchvision import transforms

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(512, 1)

    def forward(self, k, q):
        bs = q.shape[0]
        x = k * q
        x = self.fc1(x)
        out = torch.max(x, dim=1).values
        return out

def process_video(x, preprocess):
    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')
    start = datetime.now()
    name = str(df['videoid'][x])
    vid_tensor = None
    try:
        video = cv2.VideoCapture(df['contentUrl'][x])
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for fno in range(0, length):
            video.set(cv2.CAP_PROP_POS_FRAMES, fno)
            _, cvImg = video.read()
            cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
            pilImg = Image.fromarray(cvImg)
            tensorImg = preprocess(pilImg).unsqueeze(0)
            if vid_tensor is None:
                vid_tensor = tensorImg
            else:
                vid_tensor = torch.cat((vid_tensor, tensorImg))
    except Exception as e:
        print('Video %s encountered exception %s when extracting...' % (name, e))
        return

    total_time = datetime.now() - start
    
    return total_time.microseconds


def process_video1(x, preprocess):
    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')
    start = datetime.now()
    name = str(df['videoid'][x])
    vid_tensor = None
    try:
        video = cv2.VideoCapture(df['contentUrl'][x])
        frames_tensors=[]
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_tensors.append(frame)
        frames_tensor = np.asarray_chkfinite(frames_tensors, dtype=np.float32)
        frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False)
        frames_tensor = preprocess(frames_tensor)
    

        # video = cv2.VideoCapture(df['contentUrl'][x])
        # fps = video.get(cv2.CAP_PROP_FPS)
        # length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # for fno in range(0, length):
        #     video.set(cv2.CAP_PROP_POS_FRAMES, fno)
        #     _, cvImg = video.read()
        #     cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        #     pilImg = Image.fromarray(cvImg)
        #     tensorImg = preprocess(pilImg).unsqueeze(0)
        #     if vid_tensor is None:
        #         vid_tensor = tensorImg
        #     else:
        #         vid_tensor = torch.cat((vid_tensor, tensorImg))
    except Exception as e:
        print('Video %s encountered exception %s when extracting...' % (name, e))
        return

    total_time = datetime.now() - start
    
    return total_time.microseconds
    


def main():
    # ckpt = torch.load("inquire/ckpt/2023-12-06_18-39-17/VALaccFFN.pt")
    # # Print model structure
    # for key in ckpt.keys():
    #     print(key)
    #     print(ckpt[key].shape)
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    preprocess1 = transforms.Compose([transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True), transforms.CenterCrop(size=(224, 224)), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    
    ori = 0
    mod = 0
    for x in range(100):
        ori += process_video(x, preprocess)
        mod += process_video1(x, preprocess1)
        print(ori,mod)
    print(ori/100)
    print(mod/100)

if __name__ == '__main__':
    main()