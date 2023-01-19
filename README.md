# light pose
人体关键点检测

## 项目介绍    
人体关键点检测  

![video](https://cdn.staticaly.com/gh/essential-gx/img@main/test/light_pose_01.gif)

![video2](https://cdn.staticaly.com/gh/essential-gx/img@main/test/s1.gif)

## 项目配置   
* 作者开发环境：   
* Python 3.7   
* PyTorch >= 1.5.1  

## 数据集     
### coco2017 数据集  
* coco 数据集官方网站：https://cocodataset.org/#home
* [数据集下载地址(百度网盘 Password: nvr0 )](https://pan.baidu.com/s/1k7W07zXY97ueLanYRG3-cw)  ，数据集如有侵权请联系删除。

## 预训练模型   
* [预训练模型下载地址(百度网盘 Password: fztu )](https://pan.baidu.com/s/1cC8_oEVmVpKac5f_dK5spQ)   

## 项目使用方法  
* step 1: python prepare_train_labels.py  
* step 2: python make_val_subset.py  
* step 3: python train.py   # 训练
* step 4: python inference_video.py   # 视频模型推理
