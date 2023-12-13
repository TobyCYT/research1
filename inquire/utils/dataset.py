import os
from tqdm import tqdm

import torch

#create a dataset that contains the video features and the labels

# class VideoDataset(torch.utils.data.Dataset):
#     def __init__(self, mode = 'train', transform=None):
#         self.labels = []
#         self.videos = []
#         self.mode = mode
#         self.transform = transform

#         if mode == 'train':
#             for file in tqdm(os.listdir('dataset/c4c/train/labels'), total=len(os.listdir('dataset/c4c/train/labels')), desc='Loading videos'):
#                 try:
#                     vid_id = int(file.split('.')[0])
#                 except:
#                     continue
#                 vid_fea = torch.load('dataset/c4c/train/videos/%s.pt'%vid_id)
#                 vid_label = torch.load('dataset/c4c/train/labels/%s.pt'%vid_id)
#                 self.labels.append(vid_label)
#                 self.videos.append(vid_fea)
#         elif mode == 'val':
#             for file in tqdm(os.listdir('dataset/c4c/val/labels'), total=len(os.listdir('dataset/c4c/val/labels')), desc='Loading videos'):
#                 try:
#                     vid_id = int(file.split('.')[0])
#                 except:
#                     continue
#                 vid_fea = torch.load('dataset/c4c/val/videos/%s.pt'%vid_id)
#                 vid_label = torch.load('dataset/c4c/val/labels/%s.pt'%vid_id)
#                 self.labels.append(vid_label)
#                 self.videos.append(vid_fea)

#     def __len__(self):
#         return len(self.videos)

#     def __getitem__(self, idx, size=10):
#         video = self.videos[idx]
#         label = self.labels[idx]
#         sample_label = None
#         # 50/50 chance to select the same label using torch
#         if torch.rand(1) > 0.5:
#             # Randomly generate 10 indexes and select the videos with the indexes
#             rand_idx = torch.randint(len(self.videos), (size,))
#             sample = [self.videos[x] for x in rand_idx]
#             sample_label = False

#         else:
#             # Randomly generate 9 indexes and select the videos with the indexes
#             rand_idx = torch.randint(len(self.videos), (size-1,))
#             sample = [self.videos[x] for x in rand_idx]
#             sample.append(video)
#             sample_label = True

#         with torch.no_grad():
#             sample = torch.stack(sample)

#         if self.transform is not None:
#             sample = self.transform(sample)

#         return sample, label, sample_label
    
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', transform=None):
        self.labels = []
        self.videos = []
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            for file in tqdm(os.listdir('dataset/c4c/train/retrieve'), total=len(os.listdir('dataset/c4c/train/retrieve')), desc='Loading videos'):
                with torch.no_grad():
                    self.videos.append(torch.load('dataset/c4c/train/retrieve/%s'%file))
                    self.labels.append(int(file.split('.')[0].split('_')[1]))
        elif mode == 'val':
            for file in tqdm(os.listdir('dataset/c4c/val/retrieve'), total=len(os.listdir('dataset/c4c/val/retrieve')), desc='Loading videos'):
                with torch.no_grad():
                    self.videos.append(torch.load('dataset/c4c/val/retrieve/%s'%file))
                    self.labels.append(int(file.split('.')[0].split('_')[1]))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx][1:].detach()
        video.requires_grad = False
        label = self.videos[idx][0].detach().unsqueeze(0)
        label.requires_grad = False
        sample_label = self.labels[idx]
        
        return video, label, sample_label


def main():
    # test the dataset
    dataset = VideoDataset(mode='train')
    print(len(dataset))
    sample, label, sample_label = dataset[0]
    print(sample.shape, label.shape, sample_label)


if __name__ == '__main__':
    main()