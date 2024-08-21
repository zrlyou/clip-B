第四届计图人工智能挑战赛-开放域少样本视觉分类赛题-B榜代码

环境配置：
- ubuntu 18.04 LTS
- python >= 3.7
- jittor >= 1.3.0
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

方法的详细思路
1.构建训练数据集和验证数据集：
在官方提供的训练数据集中，对于每个类别，从数据集中挑出任意4张图片训练模型，共1496张图片，构成模型训练集；剩余180233张图片作为模型验证集，用验证集中所有的数据对模型进行测试；

2.创建AdanBelief优化器：
该项目使用了自定义优化器AdanBelief，在计图架构根目录下的optim.py文件中，自己编写优化器AdanBelief，该优化器是Adan优化器和AdaBelief优化器的融合，在Adan优化器中融入"Belief"增强训练模型的泛化性能，
所以要用开源的optim.py文件替换计图架构根目录下的optim.py文件；

3.利用AdanBelief优化器训练ViT-B/32版本的CLIP模型
首先冻结OpenAI官方预训练的ViT-B/32版本的CLIP模型中的全部图像层，再利用AdanBelief优化器训练模型，训练300个epoch，每隔5个epoch进行对模型进行保存，模型保存在/ckptFE/中，具体训练参数可参考train_clip.py；

4.验证模型精度
当完成CLIP模型训练后，运行test_clip.py，用验证集中所有的数据和自定义的提示词对保存的模型(['min_loss', 20, 50, 70, 90, 100, 150, 200, 250, 300])进行测试，测试结果保存在/ckptFE/test.log中；

5.测试模型精度
选取验证精度最好的模型和对应的提示词，运行test.py文件，在官方给定的数据中进行测试，输出"result.txt"。开放域少样本视觉分类赛题-B榜选择“epoch_90.pth”模型和“1. basic: a photo of”提示词，提交官方系统测试，top1的精度是0.7103。

使用的预训练模型种类：ViT-B/32版本的CLIP模型的Jittor版（https://github.com/uyzhang/JCLIP）

最终的参数量之和：151.28M

qq：279861399

A榜GitLink开源链接：https://www.gitlink.org.cn/BIT2024/clip/tree/master
A榜GitHub开源链接：https://github.com/zrlyou/clip