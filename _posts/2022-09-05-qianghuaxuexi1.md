---
layout: post
title: 高级强化学习——马尔可夫决策过程
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

只列举了一些新学的，ppt上的内容就不抄过来了。

1. 随机过程和概率论的区别：是否有时间的概念。
2. 为什么需要折扣因子$$\gamma$$ ：
   1. 类似钱会贬值，即时奖励收益更大。
   2. 长期的奖励不确定性大。
   3. 如果是一个无限的随机过程，折扣因子会让回报有一个上界，不会无限增殖下去。
   4. 如果更关注即时奖励$$\gamma \rightarrow 0$$，如果更关注长期奖励$$\gamma \rightarrow 1$$

3. 状态价值函数和动作价值函数的区别在于动作价值函数中包含了动作。
4. 策略的状态访问分布：$$v^{\pi}(s)$$，指任意时刻到达状态s的概率。
5. 对策略的价值评估$$V^{\pi}(s)$$步骤：①算策略的价值②评估价值，更新策略$$\pi$$，重复两个步骤
   1. 问题是策略收敛了但是价值不收敛，耗时

6. 有模型的方法是指已知了状态转移概率。