import os
import random
import time
import jittor as jt
from PIL import Image
from jittor.dataset import Dataset
import jittor.transform as T
import jclip as clip
from path_config import DATASET_ROOT_DIR


jt.flags.use_cuda = 1

SEED = 123

# 训练数据集
class CLIPDataset(Dataset):
    """训练数据集"""
    def __init__(self, 
                 img_list, 
                 txt_list, 
                 label_list, 
                 transform,
                 batch_size,
                 num_workers,
                 shuffle,
                 return_desc
                 ):
        super(CLIPDataset, self).__init__()
        self.image_path = img_list
        self.label_list = label_list
        self.texts  = clip.tokenize(txt_list).numpy()
        self.transform = transform
        self.return_desc = return_desc
        self.set_attrs(batch_size=batch_size,
                       num_workers=num_workers,
                       shuffle=shuffle)
        
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_path[idx]))
        text = self.texts[idx]
        label = self.label_list[idx]
        if self.return_desc:
            return image, label, text
        return image, label
    

def get_map_dict():
    """映射：从类别（数字）到文本"""
    with open(os.path.join(DATASET_ROOT_DIR, 'classes_b.txt'), 'r') as file:
        classes = file.read().splitlines()
    map_dict = {}
    id_list = [int(line.split(' ')[1]) for line in classes]
    classnames = [line.split(' ')[0] for line in classes]
    for idx, name in zip(id_list, classnames):
        if idx < 52:
            map_dict[idx] = ' '.join(name[7:].lower().split('_'))
        elif idx < 143:
            map_dict[idx] = ' '.join(name[12:].lower().split('_'))
        elif idx < 244:
            map_dict[idx] = ' '.join(name[9:].lower().split('_'))
        else:
            map_dict[idx] = ' '.join(name[8:].lower().split('_'))
    return map_dict
   


def get_description(id_list, version, custom_desc=None):
    """根据类别id获取对应的文本描述"""
    id2cls = get_map_dict()
    if version == 1:
        return ['a photo of {}'.format(id2cls[idx]) for idx in id_list]
    elif version == 2:
        desc = []
        for idx in id_list:
            if idx < 52:
                desc.append('a photo of {}, a type of animal'.format(id2cls[idx]))
            elif idx < 143:
                desc.append('a photo of {}, a type of caltech'.format(id2cls[idx]))
            elif idx < 244:
                desc.append('a photo of {}, a type of food'.format(id2cls[idx]))
            else:
                desc.append('a photo of {}, a type of dog'.format(id2cls[idx]))
        return desc
    elif version == 3:
        assert custom_desc is not None, 'custon_desc must not be None'
        return custom_desc
    raise ValueError("version must be 1 or 2 or 3")

    
def split_data(seed, version, custom_desc=None):
    """划分数据集，每个类别4张作为训练，其他用作验证"""
    # Load the training data
    with open(os.path.join(DATASET_ROOT_DIR, 'train.txt'), 'r') as file:
        img_label_pairs = file.read().splitlines()
    random.seed(seed)
    random.shuffle(img_label_pairs)


    total_paths = [l.split(' ')[0] for l in img_label_pairs]
    total_labels = [int(l.split(' ')[1]) for l in img_label_pairs]

    cnt = {}
    train_paths = []
    train_labels = []

    # animal, caltech, food, dogs
    test_paths = [[] for _ in range(4)]
    test_labels = [[] for _ in range(4)]

    for path, label in zip(total_paths, total_labels):
        if label not in cnt:
            cnt[label] = 0
        if cnt[label] < 4:
            train_paths.append(f'{DATASET_ROOT_DIR}/{path}')
            train_labels.append(label)
            cnt[label] += 1
        else:
            if label < 52:
                index = 0
            elif label < 143:
                index = 1
            elif label < 244:
                index = 2
            else:
                index = 3
            test_paths[index].append(f'{DATASET_ROOT_DIR}/{path}')
            test_labels[index].append(label)

    train_text_desc = get_description(train_labels, version, custom_desc)
        
    return {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'train_text_desc': train_text_desc,
        'test_paths': test_paths,
        'test_labels': test_labels
    }

def get_dataloader(transform, batch_size, num_workers, shuffle=True, version=2):
    data = split_data(SEED, version=version)
    return CLIPDataset(data['train_paths'], 
                            data['train_text_desc'], 
                            data['train_labels'],
                            transform,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            return_desc=True)


if __name__ == "__main__":
    
    SEED = 123

    # load data only to find data read bottleneck
    print('loading model...')
    model, preprocess = clip.load("pretrained/ViT-B-32.pkl")
    print('model loaded!')
    
    data = split_data(SEED, version=2)
   
    # preview descriptions 
    for item in data['train_text_desc']:
        print(item)
    
    train_loader = CLIPDataset(data['train_paths'], 
                               data['train_text_desc'], 
                               data['train_labels'],
                               preprocess,
                               batch_size=128,
                               num_workers=8,
                               shuffle=True,
                               return_desc=True)
                               
    
    times = []
    for epoch in range(10):
        s = time.time()
        for i, (img, text, label) in enumerate(train_loader):
            print(img.shape, text.shape, label.shape)
        e = time.time()
        times.append(e - s)
        print(f'cost time: {e - s}')
    
    print(f'average cost time: {sum(times) / len(times)}')
    
    