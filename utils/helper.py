
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jittor as jt
from jittor import transform
from jittor.optim import Adam, AdamW, SGD, Adan, AdanBelief
from PIL import Image
from datetime import datetime
from natsort import natsorted
# from jt.optim.Optimizer import AdamwBelief

class EarlyStop(object):
    """早停
    1. 当模型的损失长时间不下降时，停止训练
    2. 当模型的损失长时间增大时，也提前停止训练
    """
    def __init__(self, patience=7, delta=0.0001, patience_up=20):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.counter_up = 0
        self.last_loss = None
        self.early_stop = False
        self.patience_up = patience_up
        
    def __call__(self, loss):
        """当输入的loss多次不下降或者上升的时候，返回True，正常时返回False

        Args:
            loss (float): 当前的损失值

        Returns:
            bool: 是否早停
        """
        if self.last_loss is None:
            self.last_loss = loss
            return False

        # loss下降明显低于delta，当前清零
        if loss < self.last_loss - self.delta:
            self.counter = 0
            self.counter_up = 0
            self.last_loss = loss
        
        # loss上升明显高于delta，counter_up开始计数
        elif loss > self.last_loss + self.delta:
            self.counter_up += 1
            if self.counter_up >= self.patience_up:
                self.early_stop = True
                return True
        
        # loss上升和下降均小于delta，在区间震荡，counter开始计数
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False
    
    
def accuracy(model, dataloader, zeroshot_weights):
    """计算模型的准确率"""
    model.eval()
    corrct = 0
    total_count = 0
    with jt.no_grad():
        for i, batch in enumerate(dataloader):
            images, targets, texts  = batch
            total_count += len(images)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            logits = (100 * image_features @ zeroshot_weights).softmax(dim=-1)
            preds = jt.argmax(logits, dim=1)[0]
            corrct += jt.equal(preds, targets).sum().item()
    return corrct / total_count

def get_current_date(end_time='day'):
    # 获取当前日期时间对象
    current_date = datetime.now()
    # 格式化日期为月日时分格式
    if end_time == 'day':
        formatted_date = current_date.strftime("%m-%d")
    elif end_time == 'minute':
        formatted_date = current_date.strftime("%m-%d_%H:%M")
    return formatted_date

def get_save_path(given_path, optimizer):
    """获取tensorboard日志/模型保存路径"""
    # 文件保存路径如下：
    # given_path/date/optimizer/version_x
    
    path = os.path.join(given_path, get_current_date(end_time='day'))
    os.makedirs(path, exist_ok=True)
    
    try:
        last_version = int(natsorted(os.listdir(path))[-1].split('_')[-1])
        current_path = os.path.join(path, f'version_{last_version + 1}')
        os.makedirs(current_path, exist_ok=True)
    except IndexError:
        current_path = os.path.join(path, 'version_0')
        os.makedirs(current_path, exist_ok=True)
    return current_path



        

def get_optimizer(args, model):
    """根据输入参数获取优化器"""
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                         betas=args.betas, eps=args.eps)
        
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                          betas=args.betas, eps=args.eps)

    elif args.optimizer == 'Adan':
        if len(args.betas) == 2:
            raise ValueError('Adan optimizer requires betas has the shape like (0.9,0.98, 0.99)')
        optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                         betas=args.betas, eps=args.eps)
    
    elif args.optimizer == 'AdanBelief':
        if len(args.betas) == 2:
            raise ValueError('AdanBelief optimizer requires betas has the shape like (0.9,0.98, 0.99)')
        optimizer = AdanBelief(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                         betas=args.betas, eps=args.eps)

    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                        momentum=0.937)
        
    else:
        raise ValueError('Unsupported optimizer, please check the optimizer name.')
    
    return optimizer

def get_scheduler(optimizer, args):
    """根据输入参数获取学习率调度器"""
    pass

def get_transform(args):
    """根据输入参数获取数据预处理"""
    if args.data_preprocess == 1:
        transforms = transform.Compose([
            transform.Resize(224, mode=Image.BICUBIC),
            transform.CenterCrop(224), lambda image: image.convert("RGB"),
            transform.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))
        ])
        return transforms
    elif args.data_preprocess == 2:
        transforms = transform.Compose([
            transform.Resize(224, mode=Image.BICUBIC),
            transform.CenterCrop(224), lambda image: image.convert("RGB"),
            transform.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1),
            transform.RandomRotation(10),
            transform.RandomHorizontalFlip(),
            transform.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))])
            

def compute_loss(logits_image, logits_text):
    """计算损失函数，用来建立文本与图像的语义关系，实现语义对其"""
    ground_truth = jt.arange(len(logits_image), dtype=jt.int32)
    loss = (jt.nn.cross_entropy_loss(logits_image, ground_truth) +\
        jt.nn.cross_entropy_loss(logits_text, ground_truth)) / 2
    
    # loss = (jt.nn.cross_entropy_loss(logits_image, ground_truth) +\
    #     jt.nn.smooth_l1_loss(logits_text, ground_truth)) / 2

    return loss