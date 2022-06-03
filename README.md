# 基于YOLOv5的口罩检测

## 项目概况

项目目录结构：

![image-20220603222735233](pictures\目录结构.png)

如上图所示，项目根目录为`E:/Projects/FaceMaskDetection`。项目分为两个子工程，第一个工程为dataset，第二个子工程为yolov5。子工程dataset负责处理数据集，子工程yolov5负责训练模型并输出检测结果。

项目我已开源至我的Github个人主页：https://github.com/li192863/face-mask-detection。

## 数据处理

### 准备数据集

论文中只提到了用的是公开可用的数据集，但并未给出数据集的准确地址。此处我使用了Kaggle中的口罩检测数据集，详细地址为https://www.kaggle.com/datasets/andrewmvd/face-mask-detection。该数据集每个边界框信息有三类，分别为'with_mask'（0）, 'without_mask'（1）, 'mask_weared_incorrect'（2）。

<img src="pictures\下载数据集.png" alt="image-20220603222458789" style="zoom:50%;" />

在网页中下载数据集至本地，解压至`dataset`文件夹下（此时`dataset`文件夹下只有`images`文件夹和`annotations`文件夹）。`images`文件夹下存放853张图片（`maksssksksss*.png`），`annotations`文件夹下存放853张图片的标注信息（`maksssksksss*.xml`）。每张图片与标注文件除后缀名外名称相同。

![image-20220603230956589](pictures\dataset子项目.png)

### 划分数据集

在`dataset`文件夹下创建`split_train_val.py`，代码如下：

```python
import os
import random
import argparse


def parse_args():
    """
    传入命令行参数
    :return: 参数
    """
    parser = argparse.ArgumentParser()
    # xml文件(原始注解)的地址(annotations)
    parser.add_argument('--xml_path', default='annotations', type=str, help='input xml label path')
    # txt文件(划分后文件名)的地址(image_sets)
    parser.add_argument('--txt_path', default='image_sets', type=str, help='output path of splited file names')
    # txt文件(划分后路径名)的地址(dataset_path)
    parser.add_argument('--dataset_path', default='dataset_path', type=str, help='output path of splited file paths')
    opt = parser.parse_args()
    return opt


def generate_index(xml_path, trainval_percent=0.9, train_percent=0.8):
    """
    生成训练集/验证集/测试集索引
    :param xml_path: 原始注解路径地址
    :param trainval_percent: 训练集与验证集占数据集划分比
    :param train_percent: 训练集占训练集验证集划分比
    :return: xml文件列表 训练集与验证集索引列表 训练集索引列表
    """
    total_xml = os.listdir(xml_path)
    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)

    trainval_idx = random.sample(list_index, tv)
    train_idx = random.sample(trainval_idx, tr)
    return total_xml, trainval_idx, train_idx


def generate_txt(txt_path, total_xml, trainval_idx, train_idx, prefix='', suffix=''):
    """
    生成txt文件
    :param txt_path: 划分后存放txt文件的路径地址
    :param total_xml: xml文件列表
    :param trainval_idx: 训练集与验证集索引列表
    :param train_idx: 训练集索引列表
    :param prefix: txt文件中图片名称前缀
    :param suffix: txt文件中图片名称后缀
    """
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    
    # file_trainval = open(txt_path + '/trainval.txt', 'w')
    file_test = open(txt_path + '/test.txt', 'w')
    file_train = open(txt_path + '/train.txt', 'w')
    file_val = open(txt_path + '/val.txt', 'w')

    for i in range(len(total_xml)):
        line = prefix + total_xml[i][:-4] + suffix + '\n'
        if i in trainval_idx:
            # file_trainval.write(line)
            if i in train_idx:
                file_train.write(line)
            else:
                file_val.write(line)
        else:
            file_test.write(line)

    # file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()
    

if __name__ == '__main__':
    # 传入命令行参数
    opt = parse_args()
    xml_path, txt_path, dataset_path = opt.xml_path, opt.txt_path, opt.dataset_path
    # 写入txt文件
    total_xml, trainval_idx, train_idx = generate_index(xml_path)
    generate_txt(txt_path, total_xml, trainval_idx, train_idx)
    generate_txt(dataset_path, total_xml, trainval_idx, train_idx, prefix='E:/Projects/FaceMaskDetection/dataset/images/', suffix='.png')

```

程序运行结束后，子工程出现了`image_sets`文件夹与`dataset_path`文件夹。

`image_sets`文件夹下存放`test.txt`、`train.txt`、`val.txt`，分别对应测试集、训练集、验证集，每个`.txt`文件每一行都为一个文件名。

![image-20220603224821152](pictures\image_sets文件示例.png)

`dataset_path`文件夹下存放`test.txt`、`train.txt`、`val.txt`，分别对应测试集、训练集、验证集，每个`.txt`文件每一行都为一个图片的绝对路径。

![image-20220603224954691](pictures\dataset_path文件示例.png)

### 标记数据集

在`dataset`文件夹下创建`xml_to_yolo.py`，代码如下：

```python
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

DS_CLASSES = ('with_mask', 'without_mask', 'mask_weared_incorrect')  # 数据集类型
DS_ROOT = '.'  # 数据集根目录


class TransformAnnotation(object):
    """
    转换注解类
    """
    def __init__(self, class_to_idx=None, keep_difficult=False):
        """
        构造转换注解对象
        :param class_to_idx: 字典 {类名0:0, 类名1: 1, ...}
        :param keep_difficult: 是否保留困难值
        """
        self.class_to_idx = class_to_idx or dict(zip(DS_CLASSES, range(len(DS_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, is_yolo_format=False):
        """
        转换注解生成列表
        :param target: xml文件根节点
        :param is_yolo_format: 是否为yolo格式
        :return: 列表[cls_idx, x_center, y_center, width, height] or [xmin, ymin, xmax, ymax, cla_name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj[0].text.lower().strip()
            bbox = obj[5]
            bndbox = [int(bb.text) - 1 for bb in bbox]  # [xmin, ymin, xmax, ymax]

            if is_yolo_format:
                width = int(target.find('size').find('width').text)
                height = int(target.find('size').find('height').text)

                xmin, ymin, xmax, ymax = bndbox
                class_idx = self.class_to_idx[name]
                x_center = ((xmin + xmax) / 2.0 - 1) / width
                y_center = ((ymin + ymax) / 2.0 - 1) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                res += [[class_idx, x_center, y_center, box_width, box_height]]
            else:
                res += [bndbox + [name]]
        return res

    def get_yolo_txt(self, target):
        """
        生成yolo格式的txt文本
        :param target:
        :return:
        """
        boxes = self.__call__(target, is_yolo_format=True)
        res = ''
        for i in range(len(boxes)):
            class_idx, x_center, y_center, box_width, box_height = boxes[i]
            line = str(class_idx) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(box_width) + ' ' + str(box_height)
            res += line + '\n'
        return res

    def transform_xml_to_txt(self, xml_path, txt_path):
        """
        转换xml为txt文本
        :param xml_path: 存放xml文件夹路径
        :param txt_path: 存放txt文件夹路径
        :return:
        """
        if not os.path.exists(txt_path):  # 若目录不存在则创建
            os.makedirs(txt_path)
        for dirname, _, filenames in os.walk(xml_path):
            for filename in filenames:
                target = ET.parse(os.path.join(dirname, filename)).getroot()
                txt_string = self.get_yolo_txt(target)
                with open(os.path.join(txt_path, filename[:-4] + '.txt'), 'w+') as f:
                    f.write(txt_string)


class Detection(data.Dataset):
    """
    检测类
    """
    def __init__(self, root, image_set, transform=None, target_transform=None):
        """
        构造检测对象
        :param root: 根目录
        :param image_set: 图片集类型 包括train test val
        :param transform: 图片转换
        :param target_transform: 目标转换
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(self.root, 'annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, 'images', '%s.png')
        self._imgsetpath = os.path.join(self.root, 'image_sets', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        """
        获取第index张图片
        :param index:
        :return:
        """
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        """
        获取检测图片个数
        :return:
        """
        return len(self.ids)

    def show(self, index):
        """
        显示第index张图片
        :param index:
        :return:
        """
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255, 0, 0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        img.show()


if __name__ == '__main__':
    # 生成所有文件的标签
    transform = TransformAnnotation()
    transform.transform_xml_to_txt(xml_path='./annotations', txt_path='./labels')
    # 取出训练集的一张图片查看效果
    dataset = Detection(DS_ROOT, image_set='train', target_transform=transform)
    print(len(dataset))
    img, target = dataset[3]
    print(target)
    dataset.show(3)

```

程序运行结束后，子工程中出现了`labels`文件夹。控制台输出训练数据集的长度，输出训练数据集中的第三张图片的标签，并显示第三张图片与其标注。

![image-20220603225920882](pictures\标记数据集输出结果1.png)

![image-20220603225844033](pictures\标记数据集输出结果2.png)

`labels`文件夹下存放853张图片的标注信息（`maksssksksss*.txt`）。一个`.txt`文件对应一张`.png`图片，文件每一行对应一个物体，文件每一行代表的含义为边界框类别（0/1/2）、中心点x坐标（相对于整个图片宽）、中心点y坐标（相对于整个图片高）、边界框宽（相对于整个图片宽）、边界框高（相对于整个图片高）。该格式由yolov5官方指定。

![image-20220603230158030](pictures\labels文件示例.png)

至此，`dataset`子工程完成，该工程成功划分了数据集，并将标注信息转化为yolo格式。

## 模型训练

### 准备代码

在`FaceMaskDetection`文件夹下打开终端powershell，输入如下指令：

```shell
git clone https://github.com/ultralytics/yolov5  # 克隆项目源代码
cd yolov5  # 打开子工程文件夹
conda create -n yolov5  # 创建环境名为yolov5 专门用于处理该项目
conda activate yolov5  # 切换环境为yolov5
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/help/pypi/  # 使用清华镜像安装依赖
```

`yolov5`文件夹目录如下：

![image-20220603232341599](pictures\yolov5子项目.png)

### 准备数据

在`yolov5`文件夹下创建`data/FMD.yaml`，配置信息如下：

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../dataset # dataset root dir
train: dataset_path/train.txt  # train images (relative to 'path') 613 images
val: dataset_path/val.txt  # val images (relative to 'path') 154 images
test: dataset_path/test.txt # test images (optional) 86 images

# Classes
nc: 3  # number of classes
names: ['with_mask', 'without_mask', 'mask_weared_incorrect']  # class names

```

这里`train.txt`、`val.txt`、`test.txt`中分别指明了训练集、验证集、测试集的所有图片的绝对路径。YOLOv5官方规定文件查找`images`文件夹下每一张图片时，其自动查找与`images`文件夹平级的`labels`文件夹相同文件名的`.txt`文件作为图片的对应标签。官方解释如下：

![image-20220603233849752](pictures\官方文档说明.png)

### 修改模型

在`yolov5`文件夹下修改`models/yolov5s.yaml`，配置信息如下：

```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 3  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

YOLOv5官方提供5种模型（YOLOv5n、YOLOv5s、YOLOv5m、YOLOv5l、YOLOv5x）以适应不同的应用场景，这里我选择使用YOLOv5s。

### 配置训练

在`yolov5`文件夹下修改`train.py`下的`parse_opt`函数，代码如下：

```python
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/FMD.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```

这里我选择预训练权重存放于`weights`文件夹下，数据配置为`data/FMD.yaml`，轮数为200，每个批次大小为4张图片。

### 开始训练

运行`train.py`，执行训练。训练完成后，模型训练信息会保存在`runs/train/exp`文件夹下。

PR曲线：

![PR_curve](pictures\PR_curve.png)

损失变化：

![image-20220604000849011](pictures\result.png)

## 预测图片

### 配置检测

在`yolov5`文件夹下修改`detect.py`下的`parse_opt`函数，代码如下：

```python
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/FMD.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
```

这里我选择权重为`runs/train/exp/weights/best.pt`，数据配置为`data/FMD.yaml`。

### 预测图片

运行上述代码，执行检测。检测完成后，模型检测信息会保存在`runs/detect/exp`文件夹下。

`detect.py`可以选择摄像头、图片、视频等作为输入进行检测.

```shell
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

至此，`yolov5`子工程完成，该工程成功训练了模型，并且成功检测了图片。

### 检测结果

以下为一些检测结果展示。

![image-20220604002908692](pictures\pic1.png)

![image-20220604003514098](pictures\pic2.png)

![image-20220604003131098](pictures\pic3.png)

