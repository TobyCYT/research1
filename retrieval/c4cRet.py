import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

def getSspace():
    if os.path.exists('dataset/c4c/vidTensor.pt') and os.path.exists('dataset/c4c/idTensor.pt'):
        vid_search_tensor = torch.load('dataset/c4c/vidTensor.pt')
        vid_id_tensor = torch.load('dataset/c4c/idTensor.pt')
    else:
        search_space=[]
        search_id =[]
        with tqdm(total=len(os.listdir('dataset/c4c/videos/'))) as pbar:
            for file in os.listdir('dataset/c4c/videos/'): # Videa feature
                vid_id = int(file.split('.')[0])
                vid_fea = torch.load('dataset/c4c/videos/%s.pt'%vid_id)
                search_space.append(vid_fea)
                search_id.append(vid_id)
                pbar.update(1)
        vid_search_tensor = torch.squeeze(torch.stack([x for x in search_space]))
        vid_id_tensor = torch.squeeze(torch.stack([torch.tensor(x) for x in search_id]))
        torch.save(vid_search_tensor, 'dataset/c4c/vidTensor.pt')
        torch.save(vid_id_tensor, 'dataset/c4c/idTensor.pt')

    return vid_search_tensor, vid_id_tensor

def retrieve(query):
    norm = False
    searchspace, LUT = getSspace()

    cosim = nn.CosineSimilarity(dim=1, eps=1e-08)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")

    with torch.no_grad():
        inputs = tokenizer(text=query , return_tensors="pt").to(device)
        txt_fea = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        if norm:
            txt_fea = txt_fea / txt_fea.norm(dim=-1, keepdim=True)
            searchspace = searchspace / searchspace.norm(dim=-1, keepdim=True)
        prob = cosim(txt_fea.to(device),searchspace.to(device)).cpu()
        sorted = np.argsort(np.array(prob))
        top1k = sorted[-1000:]
        tmp = []
        top1kfea = None

        top1prob = prob[sorted[-1]].item()

        for x in sorted[-1000:]:
            tmp.insert(0,[LUT[x].item(), round(prob[x].item()/top1prob,4)])
            if top1kfea is None:
                top1kfea = searchspace[x].cpu().unsqueeze(0)
            else:
                top1kfea = torch.cat((searchspace[x].cpu().unsqueeze(0), top1kfea), 0)
        top1k = tmp

        # Plot a graph using the value of top1k[x][1] and save it as a pic with current query as name
        import matplotlib.pyplot as plt

        probtop1k = [x[1] for x in top1k]

        plt.figure(figsize=(20,10))
        plt.plot(probtop1k)
        #set y axis range
        plt.ylim(0, 1)
        plt.title(query)
        plt.savefig('%s.png'%query)
        plt.close()

        print('Query:', query)

        # Get the standard deviation of probtop1k
        import statistics
        print('Std Dev:', statistics.stdev(probtop1k))

        # get range
        print('Range:', max(probtop1k)-min(probtop1k))

        # Calculate pearson's mode skewness
        from scipy.stats import skew
        print('Pearson\'s mode skewness:', skew(probtop1k))
        # Calculate pearson's median skewness
        print('Pearson\'s median skewness:', skew(probtop1k, bias=False))

        # Calculate kurtosis
        from scipy.stats import kurtosis
        print('Kurtosis:', kurtosis(probtop1k))

        for x in top1k[:10]:
            getURL(x[0])



        # top10 = sorted[-10:]
        # tmp = []
        # top10fea = None # top 10 from top1 to top10


        # for x in sorted[-10:]:
        #     tmp.insert(0,[LUT[x].item(), round(prob[x].item(),4)])
        #     if top10fea is None:
        #         top10fea = searchspace[x].cpu().unsqueeze(0)
        #     else:
        #         top10fea = torch.cat((searchspace[x].cpu().unsqueeze(0), top10fea), 0)
        # top10 = tmp
        # for x in top10:
        #     print('%10d'%x[0], x[1], x[1]/top10[0][1])
        # for x in top10:
        #     getURL(x[0])

        # agg_fea = None
        # for x in top10fea:
        #     if agg_fea is None:
        #         agg_fea = x.unsqueeze(0)
        #     else:
        #         agg_fea = torch.cat((agg_fea, x.unsqueeze(0)), 0)
        # agg_fea_txt = torch.mean(torch.cat((agg_fea, txt_fea.cpu()),0),0)

        # agg_fea = torch.mean(agg_fea, 0)

        # print('Avg of top 10 vs all')

        # prob = cosim(agg_fea.unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        # rounded = ['%6f'%round(i, 4) for i in prob]
        # print(rounded)

        # print()

        # print('Avg of top 10 + text vs all')

        # prob = cosim(agg_fea_txt.unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        # rounded = ['%6f'%round(i, 4) for i in prob]
        # print(rounded)

        # print()

        # print('Top 1 vs all')

        # prob = cosim(top10fea[0].unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        # rounded = ['%6f'%round(i, 4) for i in prob]
        # print(rounded)
        
        # print()

        # print('Top 1 + text vs all')

        # prob = cosim(torch.mean(torch.cat((top10fea[0].unsqueeze(0).to(device),txt_fea),0),0).unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        # rounded = ['%6f'%round(i, 4) for i in prob]
        # print(rounded)

        # print()

        # print('Top x + text vs all')

        # for x in top10fea:
        #     prob = cosim(torch.mean(torch.cat((x.unsqueeze(0).to(device),txt_fea),0),0).unsqueeze(0).to(device),top10fea.to(device)).cpu().tolist()
        #     rounded = ['%6f'%round(i, 4) for i in prob]
        #     print(rounded)

        # print()

        # print('Top 10 - text vs all')

        # top10fea_var = (top10fea.to(device) - txt_fea).cpu()

        # prob = cosim(top10fea_var[0].unsqueeze(0).to(device),top10fea_var.to(device)).cpu().tolist()
        # rounded = ['%6f'%round(i, 4) for i in prob]
        # print(rounded)


def getURL(id):
    print(df['contentUrl'][df.index[df['videoid'] == id][0]])

def main():
    retrieve('Wild yak eating grass on rocky mountain slope illuminated by sun against himalaya ridge and clear sky on background. woolly bovine animal in its natural habitat. ladakh, india. camera stay still')
    retrieve('animal eating grass')
    retrieve('animal')

    retrieve('Cute asian thai college student girl in university uniform playing guitar and another girl come and hug her in the back in loving friendship schoolgirl and college student girl leisure concept in hd')
    retrieve('Cute asian thai girl')
    retrieve('girl')

    retrieve('aurora')
    retrieve('battleship')

if __name__ == '__main__':
    main()