import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jittor as jt
from tqdm import tqdm
from tabulate import tabulate
from tensorboardX import SummaryWriter
from utils.logger import setup_logger
from utils.helper import (EarlyStop, get_optimizer, get_save_path, 
                         accuracy, compute_loss)
from dataset import get_dataloader
from jittor.lr_scheduler import CosineAnnealingLR
from test_clip import zeroshot_classifier
from models import load_clip

jt.flags.use_cuda = 1


class Args:
    """设置训练参数"""
    def __init__(self):
        self.seed = 123             # 随机种子        
        self.optimizer = 'AdanBelief' # 提交系统测试 自测
        self.lr = 3e-6
        self.betas = (0.8, 0.8, 0.9) # self.freeze_version = 3  #'AdanBelief' # 提交系统测试 0.7103 自测 0.7373 epoch_90.pth basic 08-16/version_0/epoch_90.pth
        self.eps = 1e-12
        self.weight_decay = 0.2

        self.batch_size = 256
        self.num_workers = 8        # 导入数据使用的线程数
        
        
        self.epochs = 300           # 训练总的轮数
        
        self.early_stop = False
        self.patience = 10           # 早停“忍耐”的次数
        self.delta = 0.0001          # 早停的阈值
        self.caption_version = 1    # 图片描述使用的版本
                                    # 1. a photo of xxx
                                    # 2. 针对数据集自定义的描述

        self.data_augment = 0       # 数据增强版本
        self.model_save_path = 'ckptFE'
        self.log_save_path = 'logs'
        
        self.compute_acc_frequency = 0  # 每隔多少个epoch计算一次训练集acc，0表示不计算
        self.use_scheduler = False      # 是否使用学习率调度策略
        
        self.save_frequency = 5   # 模型保存频率，每隔多少个epoch进行保存

        self.freeze_version = 3  #'AdanBelief' # 提交系统测试 自测


    def __str__(self):
        # 将参数转换为字典
        args_dict = self.__dict__
        # 将元组转换为列表，以便tabulate可以正确处理
        if isinstance(args_dict['betas'], tuple):
            args_dict['betas'] = list(args_dict['betas'])
        # 使用tabulate生成表格
        table = tabulate([(key, value) for key, value in args_dict.items()], 
                         tablefmt="grid", headers=["Parameter", "Value"])
        return table

def train(model, optimizer, train_loader, scheduler, args, log_dir, save_dir, logger):
    
    print('model parameters: \n', args)
    logger.info('model parameters: \n')
    logger.info(args)
    
    logger.info('\n\nStart training...')
    
    if args.early_stop:
        early_stop = EarlyStop(patience=args.patience, delta=args.delta)
    writer = SummaryWriter(log_dir=log_dir)
    
    min_loss = float('inf')
    acc = None
    zeroshot_weights = zeroshot_classifier(model)

    pbar = tqdm(range(1, args.epochs+1))
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        for images, label, texts in train_loader:
            logits_per_image, logits_per_text = model(images, texts)
            loss = compute_loss(logits_per_image, logits_per_text)
            optimizer.step(loss)
            running_loss += loss.item()
        
        if args.compute_acc_frequency != 0 and epoch % args.compute_acc_frequency == 0 :
            acc = accuracy(model, train_loader, zeroshot_weights)

        if args.use_scheduler:
            scheduler.step()
            
            
        if running_loss < min_loss:
            min_loss = running_loss
            jt.save(model.state_dict(), '{}.pth'.format(os.path.join(save_dir, 'min_loss')))
        
        if epoch % args.save_frequency == 0:
            jt.save(model.state_dict(), '{}.pth'.format(os.path.join(save_dir, 'epoch_{}'.format(epoch))))
        
        if acc is not None:
            # pbar.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}, Min Loss: {min_loss:.4f}, Acc: {acc:.4f}")
            writer.add_scalar('Train/Acc', acc, epoch)
            logger.info(f"Epoch: {epoch}, Loss: {running_loss:.4f} Min Loss: {min_loss:.4f}, Acc: {acc:.4f}")
        else:
            # pbar.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}, Min Loss: {min_loss:.4f}")
            logger.info(f"Epoch: {epoch}, Loss: {running_loss:.4f} Min Loss: {min_loss:.4f}")
        writer.add_scalar('Train/Loss', running_loss, epoch)
        
        if args.early_stop and early_stop(running_loss):
            logger.info(f'Early stop triggered..., epoch: {epoch}')
            # print(f'early stop triggered..., epoch: {epoch}')
            break
        
    pbar.close() 
    logger.info('\n\nFinish training...')
    writer.close()


def main(args):
    model, transforms = load_clip(freeze_version=args.freeze_version)
    # model, transforms = clip.load('pretrained/ViT-B-32.pkl')
    train_loader = get_dataloader(transforms, args.batch_size,
                                  args.num_workers, shuffle=True, version=args.caption_version)
    
    optimizer = get_optimizer(args, model)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    model_save_path = get_save_path(args.model_save_path, args.optimizer)
    log_save_path = get_save_path(args.log_save_path, args.optimizer)
    logger = setup_logger(log_save_path)
    
    print(f'Model will be saved at {model_save_path}')
    logger.info('Model will be saved at {}'.format(model_save_path))
    
    train(model, optimizer, train_loader, scheduler, 
          args, log_save_path, model_save_path, logger)
          
    

if __name__ == "__main__":
    args = Args()
    jt.misc.set_global_seed(args.seed)
    main(args)