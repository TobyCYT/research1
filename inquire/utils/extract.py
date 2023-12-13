import os   
import pandas as pd
from tqdm import tqdm

from transformers import CLIPTokenizer, CLIPTextModelWithProjection

import torch

def main():
    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")

    for file in tqdm(os.listdir('dataset/c4c/videos'), total=len(os.listdir('dataset/c4c/videos'))):
        try:
            vid_id = int(file.split('.')[0])
        except:
            continue
        vid_label = df[df['videoid']==vid_id]['name'].values[0]
        
        inputs = tokenizer(text=vid_label , return_tensors="pt")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # Normalize embeddings for retrieval:
        final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
        final_output = final_output.cpu().detach()

        torch.save(final_output, 'dataset/c4c/labels/%s.pt'%vid_id)
        

if __name__ == '__main__':
    main()