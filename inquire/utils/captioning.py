from generate_caption import generate_caption
from transformers import AutoProcessor, Blip2ForConditionalGeneration

import torch
import pandas as pd
import os
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



def main(dev = False):
    with torch.no_grad():
        df = pd.read_csv('./dataset/filtered_30words_6sec_train.csv')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        transform = transforms.Compose([transforms.PILToTensor()])
        # df['videoid'], df['contentUrl']
        batch = []
        for x in tqdm(range(len(df)),total=len(df), desc='Generating captions'):
            name = str(df['videoid'][x])
            if dev:
                print(df['contentUrl'][x])
            if (os.path.exists("./dataset/caption/%s.txt" % name)):
                continue  # Skip if file already exists
            try:
                vid = []
                video = cv2.VideoCapture(df['contentUrl'][x])
                fps = video.get(cv2.CAP_PROP_FPS)
                length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                adj = round(fps) // 3
                if adj == 0:
                    print(df['contentUrl'][x])
                    print(fps)
                for fno in range(0, length, round(fps) // 3):  # Convert Video to 3FPS
                    video.set(cv2.CAP_PROP_POS_FRAMES, fno)
                    _, cvImg = video.read()
                    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
                    pilImg = Image.fromarray(cvImg)
                    img = transform(pilImg)
                    vid.append(img)
                video.release()

                vid = torch.stack(vid)
                vid = processor(vid, return_tensors="pt").to(device, torch.float16)['pixel_values']

                # batch.append(vid)
                generated_ids = model.generate(vid, max_new_tokens=100)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)

                caption = ''.join(caption)

                with open("./dataset/caption/%s.txt" % name, 'w') as f:
                    f.write(caption)

            except Exception as e:
                print("Video %s encountered exception %s when extracting..." % (name, e))
                continue

            # if len(batch) == batch_size:
            #     batched_input = torch.cat(batch, dim=0)
            #     generated_ids = model.generate(batched_input, max_new_tokens=100)
            #     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            #     print(len(generated_text),len(batched_input))
            #     return
            #     count = 0
            #     for i in range(len(batch)):
            #         length = len(batch[i])
                

if __name__ == '__main__':
    main(dev = False)