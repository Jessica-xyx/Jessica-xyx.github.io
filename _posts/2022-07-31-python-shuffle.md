---
layout: post
title: python中的shuffle
categories: [深度学习]
tags: python
---

打乱列表的元素顺序
![image-20220731222036437](/assets/img/image-20220731222036437.png)

![image-20220731222216314](/assets/img/image-20220731222216314.png)

下面这个源码是在网上抄的

```
def shuffle(self, x, random=None):
    """Shuffle list x in place, and return None.
    原位打乱列表，不生成新的列表。

    Optional argument random is a 0-argument
    function returning a random float in [0.0, 1.0); 
    if it is the default None, 
    the standard random.random will be used.
	可选参数random是一个从0到参数的函数，返回[0.0,1.0)中的随机浮点；
	如果random是缺省值None，则将使用标准的random.random()。
    """

    if random is None:
        randbelow = self._randbelow
        for i in reversed(range(1, len(x))):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = randbelow(i + 1)
            x[i], x[j] = x[j], x[i]
    else:
        _int = int
        for i in reversed(range(1, len(x))):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = _int(random() * (i + 1))
            x[i], x[j] = x[j], x[i]


```

