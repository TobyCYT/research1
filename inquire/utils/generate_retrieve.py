import os

import torch

from tqdm import tqdm

def generate_retrieve():
    cosim = torch.nn.CosineSimilarity(eps=1e-6)

    # Generate search space
    search_space = []
    video_id = []
    for file in tqdm(os.listdir('dataset/c4c/train/labels/'), total=len(os.listdir('dataset/c4c/train/labels/')), desc='Loading'):
        if file.endswith('.pt'):
            x = torch.load('dataset/c4c/train/videos/' + file).squeeze()
            video_id.append(file.split('.')[0])
            search_space.append(x)
    for file in tqdm(os.listdir('dataset/c4c/val/labels/'), total=len(os.listdir('dataset/c4c/val/labels/')), desc='Loading'):
        if file.endswith('.pt'):
            x = torch.load('dataset/c4c/val/videos/' + file).squeeze()
            video_id.append(file.split('.')[0])
            search_space.append(x)

    search_space = torch.stack(search_space)

    # Cosine similarity
    for id in tqdm(video_id, total=len(video_id), desc='Searching'):
        try:
            x = torch.load('dataset/c4c/val/labels/' + id + '.pt')
        except:
            x = torch.load('dataset/c4c/train/labels/' + id + '.pt')
        
        cosimilarity = cosim(search_space, x)
        # Sort the cosimilarity
        sorted, indices = torch.sort(cosimilarity, descending=True)
        # Concat the ground truth top 10 results into a tensor
        top10 = [x.squeeze()] + [search_space[indices[i]] for i in range(10)]
        top10 = torch.stack(top10)

        label = '1' if id in [video_id[i] for i in indices[:10]] else '0'

        # Save the tensor
        torch.save(top10, 'dataset/c4c/train/retrieve/' + id + '_' + label + '.pt')

    # Recall@10 = 0.740617; 74% positive samples

def split_retrieve():
    positive = []
    negative = []
    
    for file in tqdm(os.listdir('dataset/c4c/train/retrieve/'), total=len(os.listdir('dataset/c4c/train/retrieve/')), desc='Splitting'):
        if file.endswith('1.pt'):
            positive.append(file)
        else:
            negative.append(file)
    print(len(positive))
    print(len(negative))

    # Validation set
    for _ in tqdm(range(1400), total=1400, desc='Saving'):
        os.rename('dataset/c4c/train/retrieve/' + positive[0], 'dataset/c4c/val/retrieve/' + positive[0])
        os.rename('dataset/c4c/train/retrieve/' + negative[0], 'dataset/c4c/val/retrieve/' + negative[0])
        positive.pop(0)
        negative.pop(0)

    print(len(positive))
    print(len(negative))

    # Train set
    for _ in tqdm(range(len(negative)), total=len(negative), desc='Saving'):
        positive.pop(0)
        negative.pop(0)
    
    print(len(positive))
    print(len(negative))

    # Test set (positive)
    for _ in tqdm(range(len(positive)), total=len(positive), desc='Saving'):
        os.rename('dataset/c4c/train/retrieve/' + positive[0], 'dataset/c4c/test/retrieve/' + positive[0])
        positive.pop(0)

    print(len(positive))
    print(len(negative))

def main():
    split_retrieve()

if __name__ == '__main__':
    main()