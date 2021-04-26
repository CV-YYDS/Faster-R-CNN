# Faster-R-CNN(神经网络课程作业)

## Preparation
### 1.Dependency
```
# install other dependancy
pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet
```
```
# start visdom
nohup python -m visdom.server &
```
### 2.Dataset
下载数据https://pan.baidu.com/s/1AYao-vYtHbTRN-gQajfHCw，密码7yyp  
将数据解压后放入datasets/voc目录  
数据目录结构如下：  
```
datasets
          ├── voc           
          │    ├── Annotations
          │    ├── JPEGImages
          │    └── ImageSets/Main
          │            ├── train.txt
          │            └── test.txt
          └── <其他数据集>
```
## Models and Training
### 1.simple Faster R-CNN
backbone: VGG16  
train: 在simple-faster-rcnn 路径下  
```
python train.py train
```
parameters:  
- --plot-every=n: 可视化的轮次
- --env: visdom里面的环境名称 
- --voc_data_dir: VOC数据路径
- --use-drop:是否在ROI使用dropout
- --use-Adam: 是否使用Adam
- --load-path: 预训练模型路径

### 2.Faster R-CNN based on ResNet
backbone:ResNet50  
#### 实现1：自己修改源码
simple-faster-rcnn/models/faster_rcnn_resnet50.py 是根据原有的faster_rcnn_vgg16.py进行修改的文件  
将uitls/config.py里的model修改成RESNET50，即可按照上一节的方法进行训练。

### 实现2:



### 3.Faster R-CNN based on ResNet and FPN

## References
- Simple Faster R-CNN by yunchen: [https://github.com/chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
- FPN implemention by kuangliu:[https://github.com/kuangliu/pytorch-fpn](https://github.com/kuangliu/pytorch-fpn)
- FPN by WZMIAOMIAO: [https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn)
