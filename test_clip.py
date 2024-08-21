import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import functools

from typing import List
import jittor as jt
import numpy as np
from tqdm import tqdm
from PIL import Image
import jclip as clip
from tabulate import tabulate
from template import templatesv1, templatesv2, templatesv3
from path_config import DATASET_ROOT_DIR
from utils.logger import setup_logger
from utils.color_print import yellow_print


jt.flags.use_cuda = 1


@functools.lru_cache()
def get_valid_dataset():
    dataset_path = os.path.join(DATASET_ROOT_DIR, 'valid.lst')
    img_label_pairs = open(dataset_path, 'r').read().splitlines()
    image_list = [l.split(' ')[0] for l in img_label_pairs]
    id_list = [int(l.split(' ')[1]) for l in img_label_pairs]
    return image_list, id_list


def split_valid_dataset(image_list, id_list):
    valid_dataset = {
        'animal': {
            'image_list': [],
            'id_list': []
        },
        'caltech': {
            'image_list': [],
            'id_list': []
        },
        'dogs': {
            'image_list': [],
            'id_list': []
        },
        'food': {
            'image_list': [],
            'id_list': []
        }
    }
    for image, label in zip(image_list, id_list):
        if label < 52:
            valid_dataset['animal']['image_list'].append(image)
            valid_dataset['animal']['id_list'].append(label)
        elif label < 143:
            valid_dataset['caltech']['image_list'].append(image)
            valid_dataset['caltech']['id_list'].append(label)
        elif label < 244:
            valid_dataset['food']['image_list'].append(image)
            valid_dataset['food']['id_list'].append(label)
        else:
            valid_dataset['dogs']['image_list'].append(image)
            valid_dataset['dogs']['id_list'].append(label)
        
    return valid_dataset

class TestSet(jt.dataset.Dataset):
    def __init__(self, 
                 image_list, 
                 id_list, 
                 transform,
                 batch_size=256, 
                 num_workers=8):
        super(TestSet, self).__init__()
        self.image_list = image_list
        self.id_list = id_list
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.transform = transform
        
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.image_list),
            num_workers=self.num_workers,
            buffer_size=1024 * 1024 * 1024
        )


    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label = self.id_list[idx]
        image = Image.open(f'{DATASET_ROOT_DIR}/{image_path}').convert('RGB')
        if self.transform is not None:
            image = self.transform(image) 
        img = np.asarray(image)
        return img, label
  

@functools.lru_cache()
def get_classnames() -> List[str]:
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
    return list(map_dict.values())


def zeroshot_classifier(model: clip.model.CLIP, 
                        classnames: List[str] = get_classnames(), 
                        weights_version: int = 1) -> jt.Var:
    """
    使用 CLIP 模型进行零样本分类器的构建。

    Args:
    - model (clip.model.CLIP): 加载的 CLIP 模型实例。
    - classnames (list): 包含所有类别名称的列表。
    - templates (list): 包含模板字符串的列表，用于生成每个类别的文本输入。

    Returns:
    - torch.Tensor: 形状为 (embedding_size, num_classes) 的零样本权重张量。

    注：
    - 此函数假设模型已经在 GPU 上，并且模型可以处理字符串的 tokenization 和文本嵌入。
    - 输出的张量将包含每个类别的平均嵌入向量，并进行了归一化处理。
    """
    
    if weights_version == 1:
        templates = templatesv1
    elif weights_version == 2:
        templates = templatesv2
    elif weights_version == 3:
        templates = templatesv3
    else:
        raise ValueError("weights_version must be 1, 2, or 3")
    
    model.eval()
    with jt.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc='Extracting class embeddings'):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = jt.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def evaluate(model, dataloader, zeroshot_weights, name):
    model.eval()
    corrct = 0
    total_count = 0
    with jt.no_grad():
        print(f"\nTesting on {name}")
        bar = tqdm(dataloader)
        for i, batch in enumerate(bar):
            images, targets = batch
            total_count += len(images)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            logits = (100 * image_features @ zeroshot_weights).softmax(dim=-1)
            preds = jt.argmax(logits, dim=1)[0]
            corrct += jt.equal(preds, targets).sum().item()
            bar.set_description(f'{name} : {corrct}/{total_count} Acc: {corrct / total_count:.4f}')
            bar.update(1)   
        bar.close()  
    return corrct 


def main(best_model_path, weights_version):
    model, transform = clip.load("/jittor-competiiton/pretrained/ViT-B-32.pkl")
    if best_model_path is not None:
        model.load_state_dict(jt.load(best_model_path))
        yellow_print('Loaded weights from {}'.format(best_model_path))
    else:
        yellow_print('weights not loaded! using pretrained weights')
    zeroshot_weights = zeroshot_classifier(model, weights_version=weights_version)
    
    image_list, id_list = get_valid_dataset()
    
    valid_dataset = split_valid_dataset(image_list, id_list)
    
    batch_sizes = [256, 256, 512, 512]
    num_works = [4, 4, 8, 8]
    
    animal = TestSet(valid_dataset['animal']['image_list'], 
                     valid_dataset['animal']['id_list'], 
                     transform=transform,
                     batch_size=batch_sizes[0],
                     num_workers=num_works[0])
    
    caltech = TestSet(valid_dataset['caltech']['image_list'], 
                      valid_dataset['caltech']['id_list'],
                      transform=transform,
                      batch_size=batch_sizes[1],
                      num_workers=num_works[1])
    
    food = TestSet(valid_dataset['food']['image_list'],
                        valid_dataset['food']['id_list'],
                      transform=transform,
                      batch_size=batch_sizes[2],
                      num_workers=num_works[2])
    
    dogs = TestSet(valid_dataset['dogs']['image_list'],
                        valid_dataset['dogs']['id_list'],
                      transform=transform,
                      batch_size=batch_sizes[3],
                      num_workers=num_works[3]
    )
    animal_total = len(valid_dataset['animal']['image_list'])
    caltech_total = len(valid_dataset['caltech']['image_list'])
    food_total = len(valid_dataset['food']['image_list'])
    dogs_total = len(valid_dataset['dogs']['image_list'])
    
    total = animal_total + caltech_total + food_total + dogs_total
    animal_correct = evaluate(model, animal, zeroshot_weights, 'animal')
    caltech_correct = evaluate(model, caltech, zeroshot_weights, 'caltech')
    food_correct = evaluate(model, food, zeroshot_weights, 'food')
    dogs_correct = evaluate(model, dogs, zeroshot_weights, 'dogs')
    
    correct_total = animal_correct + caltech_correct + food_correct + dogs_correct
    
    metrics = [animal_correct/ animal_total,
               caltech_correct/ caltech_total,
               food_correct/ food_total,
               dogs_correct/ dogs_total,
               correct_total / total]
    
    print('Average Acc: ', metrics[-1])
    
    return [round(acc, 4) for acc in metrics]
    
    
if __name__ == '__main__':
    # 待测试的模型路径，是一个文件夹
    model_dir = '/jittor-competiiton/ckptFE/08-16/version_0'
    logger = setup_logger(model_dir, type_='test')
    
    # 需要测试的模型文件名， 如min_loss, 20, 30
    model_name = ['min_loss', 20, 50, 70, 90, 100, 150, 200, 250, 300]
    
    # 需要测试的提示词版本
    # 1. basic: a photo of 
    # 2. custom:
    # 3. from imagenet
    test_weights_version = [1, 2, 3]
    
    table_header = ['Epoch', '提示词', 'Animal', 'Caltech', 'Food', 'Dogs', 'Total']
    table_data = []
    promot = {
        1: 'basic',
        2: 'custom',
        3: 'imagenet'
    }

    for epoch in model_name:
        if isinstance(epoch, str):
            model_path = os.path.join(model_dir, 'min_loss.pth')
        elif isinstance(epoch, int):
            model_path = os.path.join(model_dir, f'epoch_{epoch}.pth')
        print(f'Testing with {model_path}')
        logger.info(f'Testing with {model_path}')
        for weights_version in test_weights_version:
            metrics = main(model_path, weights_version)
            metrics.insert(0, promot[weights_version])
            metrics.insert(0, epoch)
            table_data.append(metrics)
            print(tabulate(table_data, headers=table_header, tablefmt='fancy_grid'))
            logger.info(tabulate(table_data, headers=table_header, tablefmt='fancy_grid'))