from utils.dataset import VideoDataset

from torch.utils.data import DataLoader

# define val loader and train loader
def get_loader(mode='train', batch_size=8, num_workers=1, pin_memory=True, drop_last=True):
    if mode == 'train':
        dataset = VideoDataset(mode='train')
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=True)
    elif mode == 'val':
        dataset = VideoDataset(mode='val')
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=True)
    else:
        raise ValueError('mode must be one of [train, val]')

    return data_loader

def main():
    # test the dataloaders
    train_loader = get_loader(mode='train')
    val_loader = get_loader(mode='val')
    for i, (sample, label, sample_label) in enumerate(train_loader):
        print(sample.shape)
        print(label.shape)
        print(sample_label.shape)
        break
    for i, (sample, label, sample_label) in enumerate(val_loader):
        print(sample.shape)
        print(label.shape)
        print(sample_label.shape)
        break

if __name__ == '__main__':
    main()