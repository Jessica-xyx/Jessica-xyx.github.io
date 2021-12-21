---
layout : post
title : Asynchronous Reception of 2 RFID Tags   
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
| 论文题目     | Asynchronous Reception of 2 RFID Tags                        |
| ------------ | ------------------------------------------------------------ |
| 作者及单位   | Konstantinos Skyvalakis and Aggelos Bletsas , Senior Member, IEEE |
| 论文出处     | IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 69, NO. 8, AUGUST 2021 |
| 方法及创新点 |                                                              |
| 相关结果     |                                                              |
| 收获         |                                                              |

<a href="https://www.aliyundrive.com/s/FcRoVaZUYWt" class="external" target="_blank">点击这里查看论文（阿里云盘）</a>

#### 名词解释

* tag singulation ： The ability to encode or read an RFID tag without interfering with a tag nearby.
* 标签分离：在不干扰附近标签的情况下编码或读取特定RFID标签的能力。
* EPCglobal Class1 Gen2：（简称C1G2或Gen2）标准，是EPCglobal从2003年开始研究的第二代超高频RFID核心标准。Gen2规定了由终端用户设定的硬件产品的空中接口性能，是RFID、Internet和EPC标识组成的EPCglobal的网络基础。Gen2协议标准具有更安全的密码保护机制，它的32位密码保护比Gen1协议标准的8 位密码更安全。在管理性能方面，Gen2的超高频工作频段为860-960MHz，适合欧洲、北美、亚洲等国家和地区的无线电管理规定，为RFID的射频通信适应不同国家与地区的无线电管理创造了全球范围的应用环境条件。



## Abstract

本文主要解决两个标签的碰撞信号检测的问题。提出了在物理层的vertibi序列检测器和两符号联合标签信息检测器。与以往研究不同的是，提出的封闭系统模型在异步层面考虑了两个碰撞标签的响应，这在商业里面（Gen 2 协议）没考虑过。异步指两个标签的响应里最开始的时移$$\tau$$。

## Introduction

RFID标签正在快速发展，标签分离的研究也促进了商业的发展。讲了从最开始的帧时隙ALOHA是怎么做的，优点以及缺点，简单分析了一下碰撞概率等。然后就是信号检测的几个以前方法的介绍，包括了文献[4]({% post_url 2021-12-14-Single Antenna Physical Layer Collision Recovery Receivers for RFID Readers %})的。然后就讲自己的方法的重点：提出了一个封闭系统模型，考虑两个信号的时移来做碰撞检测，设计了一个shaping matrix，设计使用了两符号联合检测器和维特比联合序列检测器。以及一个符号说明。

## 系统模型

本文用的是RFID的Monostatic模型 

![image-20211220232905900](/assets/img/image-20211220232905900.png)

大规模通道路径损耗模型和小规模莱斯flat衰弱信道模型（留疑待查）

![image-20211221003613993](/assets/img/image-20211221003613993.png)

![image-20211221003714632](/assets/img/image-20211221003714632.png)

## 信号模型

![image-20211221003812885](/assets/img/image-20211221003812885.png)

![image-20211221003833881](/assets/img/image-20211221003833881.png)

![image-20211221003926361](/assets/img/image-20211221003926361.png)

公式7到公式8的过程👇

![QQ20211221003242](/assets/img/QQ20211221003242.png)

**说白了系统模型和信号模型里这么一堆公式都是为了推出公式8**



## 问题转换

信号对应关系

![image-20211221153305944](/assets/img/image-20211221153305944.png)

Table1 的上图的推导过程

![QQ20211221152631](/assets/img/QQ20211221152631.png)

![QQ20211221152814](/assets/img/QQ20211221152814.png)

信号的能量计算公式：

![QQ20211221160635](/assets/img/QQ20211221160635.jpg)

## 检测技术

维特比前向传播计算过程：

![QQ20211221214833](/assets/img/QQ20211221214833.png)

## DIGITAL LINK HOUSEKEEPING







