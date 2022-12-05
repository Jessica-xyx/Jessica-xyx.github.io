---
layout : post
title : 实信号与复信号
categories: [信号]
tags: 
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
# 相关资料

## 傅里叶变换

傅里叶变换的目的是可将时域（即时间域）上的信号转变为频域（即频率域）上的信号。

傅里叶变换公式为：

$$
F(\omega)=\mathcal{F}[f(t)]=\int_{-\infty}^{\infty} f(t) e^{-i w t} d t
$$

其中，$$\omega$$代表频率，$$t$$代表时间，$$e^-i\omega t$$是复变函数。

## 卷积定理

* 傅里叶变换的重要性质

* 卷积定理：函数卷积的傅里叶变换是函数傅里叶变换的卷积。

$$
f_{1}(t) \leftrightarrow F_{1}(\omega) ， f_{2}(t) \leftrightarrow F_{2}(\omega)
$$

$$\leftrightarrow$$代表傅里叶变换。

* 时域卷积定理：时域内的卷积对应于频域内的乘积。

$$
F\left[f_{1}(t) * f_{2}(t)\right]=F_{1}(\omega) \bullet F_{2}(\omega)
$$

$$F$$代表傅里叶变换。

* 频域卷积定理：频域内的卷积对应于时域内的乘积。两信号在时域的乘积对应于这两个信号频域的卷积除以$$2\pi$$。

$$
F\left[f_{1}(t) \bullet f_{2}(t)\right]=\frac{1}{2 \pi} F_{1}(\omega) * F_{2}(\omega)
$$

# 实信号与复信号

实信号：**$$f(t)$$是时间t的实函数。**物理可实现的信号常常是时间$$t$$（或$$k$$）的实函数（或序列），其在各时刻的函数（或序列）值为实数，这样的信号称为实信号。

对实信号$$f(t)$$做傅里叶变换得：
$$
\begin{aligned}
\mathcal{F}(\mathrm{jw}) &=\int_{-\infty}^{\infty} f(t) e^{-j w t} d t=\int_{-\infty}^{\infty} f(t) \cos (w t) d t-j \int_{-\infty}^{\infty} f(t) \sin (w t) d t \\
&=R(w)+j X(w)=|\mathcal{F}(\mathrm{jw})| e^{j \varphi(w)}
\end{aligned}
$$

实信号频谱具有共轭对称性：
$$
\mathcal{F}(\mathrm{jw})=\int_{-\infty}^{\infty} f(t) e^{-j w t} d t=\left[\int_{-\infty}^{\infty} f(t) e^{j w t} d t\right]^{*}=\mathcal{F}^{*}(-j w)
$$
频谱函数的实部和虚部分别为：**其中实部是偶函数，虚部是奇函数。**
$$
\begin{aligned}
&R(w)=\int_{-\infty}^{\infty} f(t) \cos (w t) d t \\
&X(w)=-\int_{-\infty}^{\infty} f(t) \sin (w t) d t
\end{aligned}
$$
模（幅度）和相角分别为：**其中幅度是偶函数，相角是奇函数。**
$$
\begin{aligned}
&|\mathcal{F}(\mathrm{jw})|=\sqrt{R^{2}(w)+X^{2}(w)} \\
&\varphi(w)=\arctan \left[\frac{X(w)}{R(w)}\right]
\end{aligned}
$$
**实信号具有共轭对称的频谱，从信息的角度来看，其负频谱部分是冗余的，因此为了信号处理方便，去掉频域的负半平面，只保留正频谱部分的信号，其频谱不存在共轭对称性，这样产生的频谱所对应的时域信号就是一个复信号，这个复信号称为解析信号或预包络。**