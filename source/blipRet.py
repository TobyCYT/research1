import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from lavis.models import load_model_and_preprocess

df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

missing = [1621, 2397, 2477, 2488, 2489, 4379, 12192, 13058, 15517, 15518, 23377, 23451, 23491, 23514, 25831, 25836, 25837, 2005, 2013, 5535, 5622, 7828, 7835, 7878, 11099, 12690, 12693, 28138]

def getSspace():
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

def retrieve(query):
    searchspace, LUT = getSspace()

    cosim = nn.CosineSimilarity(dim=1, eps=1e-08)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

    with torch.no_grad():
        txt_fea = model.extract_features({"text_input": [txt_processors["eval"](query)]}, mode="text").text_embeds_proj[:,0,:]
        print(txt_fea)
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

        agg_fea = None
        for x in top10fea:
            if agg_fea is None:
                agg_fea = x.unsqueeze(0)
            else:
                agg_fea = torch.cat((agg_fea, x.unsqueeze(0)), 0)
        agg_fea_txt = torch.mean(torch.cat((agg_fea, txt_fea.cpu()),0),0)

        agg_fea = torch.mean(agg_fea, 0)

        print('Avg of top 10 vs all')

        prob = cosim(agg_fea.unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        rounded = ['%6f'%round(i, 4) for i in prob]
        print(rounded)

        print()

        print('Avg of top 10 + text vs all')

        prob = cosim(agg_fea_txt.unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        rounded = ['%6f'%round(i, 4) for i in prob]
        print(rounded)

        print()

        print('Top 1 vs all')

        prob = cosim(top10fea[0].unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        rounded = ['%6f'%round(i, 4) for i in prob]
        print(rounded)
        
        print()

        print('Top 1 + text vs all')

        prob = cosim(torch.mean(torch.cat((top10fea[0].unsqueeze(0).to(device),txt_fea),0),0).unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        rounded = ['%6f'%round(i, 4) for i in prob]
        print(rounded)

        print()

        print('Top x + text vs all')

        for x in top10fea:
            prob = cosim(torch.mean(torch.cat((x.unsqueeze(0).to(device),txt_fea),0),0).unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
            rounded = ['%6f'%round(i, 4) for i in prob]
            print(rounded)

        print()

        print('Top 10 - text vs all')

        top10fea_var = (top10fea.to(device) - txt_fea).cpu()

        prob = cosim(top10fea_var[0].unsqueeze(0).to(device),top10fea_var.to(device)).cpu().tolist()
        rounded = ['%6f'%round(i, 4) for i in prob]
        print(rounded)



def getURL(id):
    print(df['contentUrl'][df.index[df['videoid'] == id][0]])

def main():
    retrieve('a man walking his dog')

if __name__ == '__main__':
    main()