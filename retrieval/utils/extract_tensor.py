import pandas as pd
import os
import torch
import cv2
import clip
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
import threading
from concurrent.futures import ThreadPoolExecutor

# blip missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837]
# clip missing = [1621, 2005, 2013, 2397, 2477, 2488, 2489, 4379, 5535, 5622, 7828, 7835, 7878, 11099, 12192, 12690, 12693, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 28138]
# total missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 2005, 2013, 5535, 5622, 7828, 7835, 7878, 11099, 12690, 12693, 28138]


def process_video(x, model, preprocess, device):
    name = str(df['videoid'][x])
    vid_tensor = None
    # if x in [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837]: # skip missing blip fea
    #     return
    if (os.path.exists("dataset/tensor/%s.pt" % name)):
        return  # Skip if file already exists
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
    
    # features_image = model.encode_image(vid_tensor.to(device)).cpu().detach()
    torch.save(vid_tensor.cpu().detach(), 'dataset/tensor/%s.pt' % name)

def run_task(df, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    with torch.no_grad():
        # for x in tqdm(df.index):
        #     process_video(x, model, preprocess, device)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = list(tqdm(executor.map(process_video, df.index, [model]*len(df.index), [preprocess]*len(df.index), [device]*len(df.index)), total=len(df.index), desc='Processing'))

if __name__ == "__main__":

    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

    # split data in half, process half in cuda:0 and half in cuda:1
    df1 = df.iloc[:len(df)//2]
    df2 = df.iloc[len(df)//2:]
    # run in parallel
    import threading
    t1 = threading.Thread(target=run_task, args=(df1, 'cuda:0'))
    t2 = threading.Thread(target=run_task, args=(df2, 'cuda:1'))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print('done')
