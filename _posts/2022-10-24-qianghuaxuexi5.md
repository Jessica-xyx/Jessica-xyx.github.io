---
layout: post
title: 高级强化学习——深度强化学习
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  ![photo]({{site.url}}/assets/img/深度强化学习.png)

学习dueling DQN有感：往往很有效的办法的核心思想是最简单的，dueling DQN其实就是一点概率上的想法，把要估计的Q拆成了均值V和优势函数，但是带来的效果却非常好，该作者也获得了ICML best student paper。

这个思想大家一看都懂了QAQ可是自己就没办法写出来，数学还是很重要啊！！resnet也是，要说这东西很难吗？但是怎么才能从曾经学过的东西运用出来实在是太难了，大部分知识学完就忘了QAQ，而且对从事的应用的理解也没有很深，也不会提关键的问题，dueling DQN是觉得好多动作没必要更新才想出个这么个东西，“很多动作没必要更新”这个问题，也是很难提出来的。

太菜了哈哈哈