---
layout: post
title: linux服务器无root权限安装pip和本地安装pytorch
categories: [linux]
tags: 
---

# linux服务器没有root权限不能用apt命令安装pip
1. ```
   vim get-pip.py
   ```

2. ```
   https://bootstrap.pypa.io/get-pip.py
   ```

   复制所有文字到新建的文件get-pip.py中

4. ```
   python get-pip.py
   ```


# 本地安装pytorch
1. 服务器直连pytorch官网报超时，所以自己下载wheels文件安装

  ```
  pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
  ```

  

2. 或者 直接将 PyTorch 安装指引 中的 

  ```
  https://download.pytorch.org/whl
  ```

   替换为

  ```
  https://mirrors.aliyun.com/pytorch-wheels
  ```

  即可，比如 

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

但是这个在阿里云里好像没有，下不成，我是用第一种方法装的，所以没法自己选cuda版本，下了个cuda10.2的，但是服务器cuda是11.6，哎，凑合用吧

![photo]({{site.url}}/assets/img/屏幕截图 2023-06-05 184557.png)



# 其他

* pip配置阿里镜像

* ```
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip config set install.trusted-host mirrors.aliyun.com
  ```

  

# 各种地址

* pytorch-wheels

* ```
  https://mirrors.aliyun.com/pytorch-wheels/
  https://pypi.org/project/torch/1.13.0/#files
  https://download.pytorch.org/whl/torch/
  ```

* pytorch

* ```
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
  但是我建议不要用这个下，不如用wheels文件
  ```

* pytorch官网的历史版本

* ```
  https://pytorch.org/get-started/previous-versions/
  ```

* anaconda

* ```
  https://repo.anaconda.com/archive/
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
  ```

  





