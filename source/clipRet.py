import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn

df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 2005, 2013, 5535, 5622, 7828, 7835, 7878, 11099, 12690, 12693, 28138]

def getSspace():
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
    searchspace, LUT = getSspace()

    cosim = nn.CosineSimilarity(dim=1, eps=1e-08)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

    with torch.no_grad():
        txt_fea = model.extract_features({"text_input": [txt_processors["eval"](query)]}, mode="text").text_embeds_proj[:,0,:]
        prob = cosim(txt_fea.to(device),searchspace.to(device)).cpu()
        sorted = np.argsort(np.array(prob))
        top10 = sorted[-10:]
        tmp = []
        for x in sorted[-10:]:
            tmp.insert(0,[LUT[x].item(), prob[x].item()])
        top10 = tmp
        print(top10)
        for x in top10:
            getURL(x[0])

def getURL(id):
    print(df['contentUrl'][df.index[df['videoid'] == id][0]])

def main():
    #retrieve('person taking picture with their phone')
    getSspace()

if __name__ == '__main__':
    main()