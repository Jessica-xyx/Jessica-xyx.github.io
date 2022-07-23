---
layout: post
title: CUDA11.4+Windows11安装pytorch
---

前期准备：安装anaconda、python、vs等

## 检查cuda和cudnn是否安装成功

```
nvcc -V
```

![image-20220723172153832](/assets/img/image-20220723172153832.png)

去cuda的安装文件

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\extras\demo_suite

在终端分别运行两个文件，如果出现以下输出表示cudnn安装成功

![image-20220723172456032](/assets/img/image-20220723172456032.png)

![image-20220723172516898](/assets/img/image-20220723172516898.png)

## 在anaconda 终端安装pytorch

```

conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```

![image-20220723174936054](/assets/img/image-20220723174936054.png)

## 在虚拟环境里使用jupyter notebook

```
conda install ipykernel

python -m ipykernel install --name pytorch
```

![image-20220723175755106](/assets/img/image-20220723175755106.png)

然后就可以在jupyter notebook里写pytorch了，不过还是推荐用vscode写:)