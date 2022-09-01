---
layout: post
title: pytorch中改变张量形状：torch.view()、torch.squeeze() / torch.unsqueeze()、torch.permute()
---

# torch.view()

* view函数使用的前提是改变前后数字的总个数不能变，（4，4）能变成（2，8），但是不能变成（3，7）

![image-20220730212727696](/assets/img/image-20220730212727696.png)

* 而且view是不会原地更改的，比如下面这样，a还是原来的a，得把view之后数的存到新的变量去

![image-20220730212346118](/assets/img/image-20220730212346118.png)

![image-20220730212536567](/assets/img/image-20220730212536567.png)

* view里也可以填-1，函数会自己计算应该变成什么样子

![image-20220730213217052](/assets/img/image-20220730213217052.png)

![image-20220730213516144](/assets/img/image-20220730213516144.png)

![image-20220730213533026](/assets/img/image-20220730213533026.png)

* 但是只能有一个-1，多了要报错

![image-20220730213251069](/assets/img/image-20220730213251069.png)

* 而且也不能让计算结果变成一个除不尽的数，比如这个4×4×4×4=256，除以3×8=24，256/24除不尽，就会报错

![image-20220730213424704](/assets/img/image-20220730213424704.png)

