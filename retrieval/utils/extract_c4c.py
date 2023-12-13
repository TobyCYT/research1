import pandas as pd
import os
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection


# missing idx = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837]

def process_video(x):
    name = str(df['videoid'][x])
    vid_tensor = None

    frame_rate=1.0
    size=224

    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),            
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)
    
    if (os.path.exists("./c4c/%s.pt" % name)):
        return  # Skip if file already exists
    
    try:
        cap = cv2.VideoCapture(df['contentUrl'][x])
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps < 1:
            images = np.zeros([3, size, size], dtype=np.float32) 
            print("ERROR: problem reading video file: ", df['videoid'][x])
        else:
            total_duration = (frameCount + fps - 1) // fps
            start_sec, end_sec = 0, total_duration
            interval = fps / frame_rate
            frames_idx = np.floor(np.arange(start_sec*fps, end_sec*fps, interval))
            ret = True     
            images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)
                
            for i, idx in enumerate(frames_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES , idx)
                ret, frame = cap.read()    
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             
                last_frame = i
                images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))
                
            images = images[:last_frame+1]
        cap.release()
        video_frames = torch.tensor(images)
        visual_output = model(video_frames.to(device))

        # Normalizing the embeddings and calculating mean between all embeddings. 
        visual_output = visual_output["image_embeds"]
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        torch.save(visual_output.cpu(), 'dataset/c4c/%s.pt' % name)

    except Exception as e:
        print('Video %s encountered exception %s when extracting...' % (name, e))
        return



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(device)

    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

    model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device)
    model = model.eval()

    process_video(1000)
    
    with ThreadPoolExecutor(max_workers=20) as executor:  # Adjust max_workers as needed
        futures = list(tqdm(executor.map(process_video, df.index), total=len(df.index), desc='Processing'))