import pandas as pd
import os
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from lavis.models import load_model_and_preprocess
from concurrent.futures import ThreadPoolExecutor

# missing idx = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837]

def process_video(x):
    name = str(df['videoid'][x])
    vid_tensor = None
    if (os.path.exists("./blip/%s.pt" % name)):
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
            tensorImg = vis_processors["eval"](pilImg).unsqueeze(0)
            if vid_tensor is None:
                vid_tensor = tensorImg
            else:
                vid_tensor = torch.cat((vid_tensor, tensorImg))
    except Exception as e:
        print('Video %s encountered exception %s when extracting...' % (name, e))
        return
    sample = {"image": vid_tensor.to(device)}
    features_image = model.extract_features(sample, mode="image")
    vid_fea = features_image.image_embeds_proj[:, 0, :]
    vid_fea = torch.mean(vid_fea, axis=0)
    torch.save(vid_fea.cpu(), 'dataset/blip/%s.pt' % name)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(device)

    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
        futures = list(tqdm(executor.map(process_video, df.index), total=len(df.index), desc='Processing'))
