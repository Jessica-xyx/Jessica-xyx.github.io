---
layout : post
title : Asynchronous Reception of 2 RFID Tags   
---
| 论文题目     | Asynchronous Reception of 2 RFID Tags                        |
| ------------ | ------------------------------------------------------------ |
| 作者及单位   | Konstantinos Skyvalakis and Aggelos Bletsas , Senior Member, IEEE |
| 论文出处     | IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 69, NO. 8, AUGUST 2021 |
| 方法及创新点 |                                                              |
| 相关结果     |                                                              |
| 收获         |                                                              |

#### 名词解释

* tag singulation ： The ability to encode or read an RFID tag without interfering with a tag nearby.
* 标签分离：在不干扰附近标签的情况下编码或读取特定RFID标签的能力。
* EPCglobal Class1 Gen2：（简称C1G2或Gen2）标准，是EPCglobal从2003年开始研究的第二代超高频RFID核心标准。Gen2规定了由终端用户设定的硬件产品的空中接口性能，是RFID、Internet和EPC标识组成的EPCglobal的网络基础。Gen2协议标准具有更安全的密码保护机制，它的32位密码保护比Gen1协议标准的8 位密码更安全。在管理性能方面，Gen2的超高频工作频段为860-960MHz，适合欧洲、北美、亚洲等国家和地区的无线电管理规定，为RFID的射频通信适应不同国家与地区的无线电管理创造了全球范围的应用环境条件。



## Abstract

本文主要解决两个标签的碰撞信号检测的问题。提出了在物理层的vertibi序列检测器和两符号联合标签信息检测器。与以往研究不同的是，提出的封闭系统模型在异步层面考虑了两个碰撞标签的响应，这在商业里面（Gen 2 协议）没考虑过。异步指两个标签的响应里最开始的时移$$\tau$$。

## Introduction

RFID标签正在快速发展，标签分离的研究也促进了商业的发展。讲了从最开始的帧时隙ALOHA是怎么做的，优点以及缺点，简单分析了一下碰撞概率等。然后就是信号检测的几个以前方法的介绍，包括了文献[4]({% post_url 2021-12-14-Single Antenna Physical Layer Collision Recovery Receivers for RFID Readers %})的。然后就讲自己的方法的重点：提出了一个封闭系统模型，考虑两个信号的时移来做碰撞检测，设计了一个shaping matrix，设计使用了两符号联合检测器和维特比联合序列检测器。以及一个符号说明。

## 系统模型

大规模通道路径损耗模型