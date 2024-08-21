import os

# 指定使用1号显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jittor as jt
from PIL import Image
import numpy as np
import jclip as clip
from tqdm import tqdm
from test_clip import zeroshot_classifier
from path_config import TESTSET_ROOT_DIR

jt.flags.use_cuda = 1

class TestSet(jt.dataset.Dataset):
    def __init__(self, 
                 image_list, 
                 transform,
                 batch_size=256, 
                 num_workers=8):
        super(TestSet, self).__init__()
        self.image_list = image_list
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.transform = transform
        
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.image_list),
            num_workers=self.num_workers
        )


    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label = image_path.split('/')[-1]
        image = Image.open(f'{TESTSET_ROOT_DIR}/{image_path}').convert('RGB')
        if self.transform is not None:
            image = self.transform(image) 
        img = np.asarray(image)
        return img, label
    
def load_model(model_path):
    model, preprocess = clip.load("/jittor-competiiton/pretrained/ViT-B-32.pkl")
    model.eval()
    model.load_state_dict(jt.load(model_path))
    return model, preprocess

def predict(model, preprocess, weights_version=1):
    # image_paths = [os.path.join(TESTSET_ROOT_DIR, img_path) for img_path in ]
    image_paths = os.listdir(TESTSET_ROOT_DIR)
    dataloader = TestSet(image_paths, preprocess )
    zeroshot_weights = zeroshot_classifier(model, weights_version=weights_version)
    result = []
    img_paths = []
    
    with jt.no_grad():
        for batch in tqdm(dataloader, desc='Preprocessing'):
            images, paths = batch
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            logits = (100 * image_features @ zeroshot_weights).softmax(dim=-1)
            _, predict_indices = jt.topk(logits, k=5, dim=1)
            img_paths += paths
            result += predict_indices.tolist()
    return result, img_paths

def main(model_path, weights_version=1):
    model, preprocess = load_model(model_path)
    result, img_paths = predict(model, preprocess, weights_version)
    out_txt(result, img_paths)

def out_txt(result, img_paths, file_name='result.txt'):
    with open(file_name, 'w') as f:
        for img, preds in zip(img_paths, result):
            f.write(f'{img} '+ ' '.join([str(i) for i in preds]) + '\n')
    print(f'Result file generated!\nFile Path: {os.path.abspath(file_name)}')
    
if __name__ == "__main__":
    # 完整的模型路径
    model_path = '/jittor-competiiton/ckptFE/08-16/version_0/epoch_90.pth'
    main(model_path, weights_version=1)
# # 提交系统测试0.7103
# # 自测 0.7373 epoch_90.pth basic