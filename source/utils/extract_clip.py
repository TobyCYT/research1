import pandas as pd
import os
import torch
import cv2
import clip
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor

# blip missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837]
# clip missing = [1621, 2005, 2013, 2397, 2477, 2488, 2489, 4379, 5535, 5622, 7828, 7835, 7878, 11099, 12192, 12690, 12693, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 28138]
# total missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 2005, 2013, 5535, 5622, 7828, 7835, 7878, 11099, 12690, 12693, 28138]


def process_video(x):
    name = str(df['videoid'][x])
    vid_tensor = None
    if x in [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837]: # skip missing blip fea
        return
    if (os.path.exists("./clip/%s.pt" % name)):
        return  # Skip if file already exists
    try:
        video = cv2.VideoCapture(df['contentUrl'][x])
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for fno in range(0, length, round(fps) // 3):  # Convert Video to 3FPS
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
    
    features_image = model.encode_image(vid_tensor.to(device))
    vid_fea = torch.mean(features_image, axis=0)
    torch.save(vid_fea.cpu(), 'dataset/clip/%s.pt' % name)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(device)

    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

    model, preprocess = clip.load("ViT-B/32", device=device)

    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
        futures = list(tqdm(executor.map(process_video, df.index), total=len(df.index), desc='Processing'))
