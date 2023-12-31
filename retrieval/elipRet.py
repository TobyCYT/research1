import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import clip
from lavis.models import load_model_and_preprocess

df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 2005, 2013, 5535, 5622, 7828, 7835, 7878, 11099, 12690, 12693, 28138]

def getBlipSS():
    if os.path.exists('dataset/blip/vidTensor.pt') and os.path.exists('dataset/blip/idTensor.pt'):
        vid_search_tensor = torch.load('dataset/blip/vidTensor.pt')
        vid_id_tensor = torch.load('dataset/blip/idTensor.pt')
    else:
        search_space=[]
        search_id =[]
        with tqdm(total=len(os.listdir('dataset/blip/'))) as pbar:
            for file in os.listdir('dataset/blip/'): # Videa feature
                vid_id = int(file.split('.')[0])
                if vid_id in [df['videoid'][x] for x in missing]:
                    print('ID of %s is not processed...' % vid_id)
                    continue
                vid_fea = torch.load('dataset/blip/%s.pt'%vid_id)
                search_space.append(vid_fea)
                search_id.append(vid_id)
                pbar.update(1)
        vid_search_tensor = torch.squeeze(torch.stack([x for x in search_space]))
        vid_id_tensor = torch.squeeze(torch.stack([torch.tensor(x) for x in search_id]))
        torch.save(vid_search_tensor, 'dataset/blip/vidTensor.pt')
        torch.save(vid_id_tensor, 'dataset/blip/idTensor.pt')

    return vid_search_tensor, vid_id_tensor

def getClipSS():
    if os.path.exists('dataset/clip/vidTensor.pt') and os.path.exists('dataset/clip/idTensor.pt'):
        vid_search_tensor = torch.load('dataset/clip/vidTensor.pt')
        vid_id_tensor = torch.load('dataset/clip/idTensor.pt')
    else:
        search_space=[]
        search_id =[]
        with tqdm(total=len(os.listdir('dataset/clip/'))) as pbar:
            for file in os.listdir('dataset/clip/'): # Videa feature
                vid_id = int(file.split('.')[0])
                if vid_id in [df['videoid'][x] for x in missing]:
                    print('ID of %s is not processed...' % vid_id)
                    continue
                vid_fea = torch.load('dataset/clip/%s.pt'%vid_id)
                search_space.append(vid_fea)
                search_id.append(vid_id)
                pbar.update(1)
        vid_search_tensor = torch.squeeze(torch.stack([x for x in search_space]))
        vid_id_tensor = torch.squeeze(torch.stack([torch.tensor(x) for x in search_id]))
        torch.save(vid_search_tensor, 'dataset/clip/vidTensor.pt')
        torch.save(vid_id_tensor, 'dataset/clip/idTensor.pt')

    return vid_search_tensor, vid_id_tensor


def retrieve(query):
    searchspaceB, BLUT = getBlipSS()
    searchspaceC, CLUT = getClipSS()
    # both LUT are same 
    searchspace = torch.cat((searchspaceB, searchspaceB,searchspaceC),1) # force 1:1 voting
    LUT = BLUT # BLUT/CLUT any will do

    cosim = nn.CosineSimilarity(dim=1, eps=1e-08)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    modelB, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    modelC, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        txt_feaB = modelB.extract_features({"text_input": [txt_processors["eval"](query)]}, mode="text").text_embeds_proj[:,0,:]
        txt_feaC = modelC.encode_text(clip.tokenize([query]).to(device))
        txt_fea = torch.cat((txt_feaB, txt_feaB, txt_feaC), 1) # 256, 512 imbalance, force 1:1 voting
        prob = cosim(txt_fea.to(device),searchspace.to(device)).cpu()
        sorted = np.argsort(np.array(prob))
        top10 = sorted[-10:]
        tmp = []
        top10fea = None # top 10 from top1 to top10
        for x in sorted[-10:]:
            tmp.insert(0,[LUT[x].item(), round(prob[x].item(),4)])
            if top10fea is None:
                top10fea = searchspace[x].cpu().unsqueeze(0)
            else:
                top10fea = torch.cat((searchspace[x].cpu().unsqueeze(0), top10fea), 0)
        top10 = tmp
        for x in top10:
            print('%10d'%x[0], x[1])
        for x in top10:
            getURL(x[0])

        for x in top10fea:
            prob = cosim(x.unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
            rounded = ['%6f'%round(i, 4) for i in prob]
            print(rounded)

def getURL(id):
    print(df['contentUrl'][df.index[df['videoid'] == id][0]])

def main():
    retrieve('a man hiking with his friends')

if __name__ == '__main__':
    main()