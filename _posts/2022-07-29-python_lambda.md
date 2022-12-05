---
layout: post
title: python中的lambda
categories: [深度学习]
tags: python
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
## 如果在短时间内需要匿名函数，请使用 lambda 函数。

**把函数写在一行里面（只能有一句话），可以接收任意多的入参。**

```
def myfunc(n):
  return lambda a, b: pow(a, n) + b

mydoubler = myfunc(2)

print(mydoubler(11,100))
```

![image-20220729225356891](/assets/img/image-20220729225356891.png)

上面这段代码的意思就是计算 $$11^2+100 = 221$$

```
def myfunc(n):
  return lambda a : pow(a, n)

mydoubler = myfunc(2)
mytripler = myfunc(3)

print(mydoubler(11)) 
print(mytripler(11))
```

![image-20220729225654189](/assets/img/image-20220729225654189.png)