from model.ffn import FFN

import os
import pandas as pd

import torch

from tqdm import tqdm


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cosim = torch.nn.CosineSimilarity(eps=1e-6)
    df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')
    
    model = FFN().to(device)
    model.load_state_dict(torch.load('inquire/ckpt/2023-12-01_18-38-46/accFFN.pt'))
    model.eval()

    # Concat all pt files in train folder in to a tensor
    search_space = []
    video_id = []
    for file in tqdm(os.listdir('dataset/c4c/train/labels/'), total=len(os.listdir('dataset/c4c/train/labels/')), desc='Loading'):
        if file.endswith('.pt'):
            x = torch.load('dataset/c4c/train/videos/' + file).squeeze()
            video_id.append(file.split('.')[0])
            search_space.append(x)
    search_space = torch.stack(search_space)

    running_corrects = 0

    for id in tqdm(video_id, total=len(video_id), desc='Searching'):
        x = torch.load('dataset/c4c/train/labels/' + id + '.pt')
        cosimilarity = cosim(search_space, x)
        # Sort the cosimilarity
        sorted, indices = torch.sort(cosimilarity, descending=True)
        # # Print ground truth
        # print('Ground truth: ' + df['contentUrl'][df.index[df['videoid'] == int(id)][0]])
        # print('Top 10 results: ')
        # # Print the id, name and link of the top 10 videos
        # for i in range(10):
        #     print(str(sorted[i].item()) + ' ' + video_id[indices[i]])
        #     print(df['contentUrl'][df.index[df['videoid'] == int(video_id[indices[i]])][0]])
        # break
        # Concat the ground truth top 10 results into a tensor
        top10 = [x.squeeze()] + [search_space[indices[i]] for i in range(10)]
        top10 = torch.stack(top10)
        # Get the prediction
        with torch.no_grad():
            top10 = top10.view(-1, 512*11)
            pred = model(top10.to(device)).cpu()
            pred = torch.sigmoid(pred)
            pred = pred > 0.5

            ground_truth = int(id) in [int(video_id[indices[i]]) for i in range(10)]
            
            # Performance matrix
            if pred == ground_truth:
                running_corrects += 1

    accuracy = running_corrects / len(search_space)
    print('Accuracy: ' + str(accuracy))

def main():
    test()

if __name__ == '__main__':
    main()