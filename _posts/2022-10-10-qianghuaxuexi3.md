---
layout: post
title: 高级强化学习——多步自助与近似逼近法
categories: [强化学习]
tags: 
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

* 蒙特卡洛要采样一整个eposide，TD算法只采样一步，为了用更多的数据，可以采用多步时序差分算法。并且步数和学习率需要平衡。
* 当状态-动作空间非常大或者是连续空间时，单纯构造Q表来维护就不太合适了，动作空间非常大会导致维数灾难，并且无法为连续空间构造一个离散的Q表。
* 解决上面这个问题有两种思路，一个是对状态/动作进行离散化或者分桶，另一个是构建参数化的值函数近似估计。
  * 对于连续状态马尔科夫决策过程，可以对状态空间离散化
  * 对于离散的情况，可以进行分桶，就是把相似的离散状态归类到一起，这样就可以减小状态空间。
* 这两种方法比较直观，操作简单，但是表示价值函数会过于简单，可能会漏掉信息，离散化的过程中也会导致维数灾难。



* 另一类办法就是构建参数化的值函数估计。也就是用可学习的函数来近似值函数，利用强化学习。优点是泛化能力强，可以在从未出现过的状态上使用。

* 近似函数期望是可微分的，便于求梯度，进行梯度下降的操作。并且强化学习和其他机器学习区别的一点在于，强化学习希望模型能在非稳态、非独立同分布的数据上训练。

* 最直接的办法就是基于随机梯度下降来近似值函数，值函数包括状态价值函数$$V_{\pi}$$和状态动作价值函数$$Q_{\pi}(s,a)$$。

  * 类似监督学习，目标函数是$J(\theta)=\mathbb{E}_\pi\left[\frac{1}{2}\left(V^\pi(s)-V_\theta(s)\right)^2\right]$，$$V^\pi(s)$$是真实值。$$V_\theta(s)$$是拟合值，接着对参数θ求导，梯度方向就是参数更新方向。
* $\begin{aligned} \theta & \leftarrow \theta-\alpha \frac{\partial J(\theta)}{\partial \theta} \\ &=\theta+\alpha\left(V^\pi(s)-V_\theta(s)\right) \frac{\partial V_\theta(s)}{\partial \theta} \end{aligned}$
  * 价值函数可以用特征的线性组合来表示$$ V_\theta(s)=\theta^{\mathrm{T}} x(s)$$
* $$V^\pi(s)$$可以用蒙特卡洛或时序差分来近似，强化学习学习的其实是一个“假目标”，并不存在真正的真实值。
  * $$Q_{\pi}(s,a)$$的估计同理

* 还有一个办法是直接学习一个策略，绕过价值函数，反正最终需要的是策略，干脆就直接对策略进行学习。因此就要参数化策略，策略可以是动作的随机概率分布。

  * 策略梯度的优化目标是$$J_\theta=E_S\left[V_\theta(s)\right]$$，V是Q的期望，$V_\theta(s)=\sum_{a \in A} \pi_\theta(a \mid s) Q_\pi(\mathrm{s}, a)$，同样可以用梯度上升策略来更新参数，梯度为$\frac{\partial J(\theta)}{\partial \theta}=\mathbb{E}_{\pi_\theta}\left[\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta} \underline{Q^{\pi_\theta}(s, a)}\right]$
  * 公式中还缺少Q值，REINFORCE方法是用蒙特卡洛来估计，用真实的回报$$G_t$$估计Q，这是一个无偏的算法。
  * REINFORCE的问题是任务必须要有终止状态，并且值函数方差较高，另一种算法Actor-Critic用一个可以训练的函数来估计REINFORCE中的$$G_t$$，因此有两个网络，一个用于估计动作价值，另一个估计动作，即策略。
  * A2C引入优势函数$A^\pi(s, a)=Q^\pi(s, a)-V^\pi(s)$来降低方差。

