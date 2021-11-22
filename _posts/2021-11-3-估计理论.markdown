---
layout: post
title: 估计理论
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
> <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# 估计理论

对未知确定性参数的估计。

 ![photo]({{site.url}}/assets/img/estimate.png)

# 无偏估计

定义：**未知参数估计量的平均值等于未知参数的真值**。

无偏估计量趋向于具有对称PDF，它的中心在真值附近。

估计量是无偏的并不意味着它就是好的估计量，这只是保证估计量的平均值为真值；另一方面，有偏估计量是由系统误差造成的一种估计，这种系统误差预先假设是不会出现的。持续不断的偏差将导致估计量的准确性变差。

# 最小方差准则

在寻找最佳估计量的时候，我们需要采用某些准则。

## 均方误差准则（mean square error）

均方误差的定义：
$$
mse(\widehat{\theta})=E[(\widehat{\theta}-\theta)^2]
$$
均方误差度量了估计量偏离真值的平方偏差的统计平均值
$$
\begin{aligned}
\operatorname{MSE}(\hat{\theta}) &=\mathrm{E}_{\theta}\left[(\hat{\theta}-\theta)^{2}\right] \\
&=\mathrm{E}_{\theta}\left[\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]+\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)^{2}\right] \\
&=\mathrm{E}_{\theta}\left[\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)^{2}+2\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)+\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)^{2}\right] \\
&=\mathrm{E}_{\theta}\left[\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)^{2}\right]+\mathrm{E}_{\theta}\left[2\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)\right]+\mathrm{E}_{\theta}\left[\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)^{2}\right] \\
&=\mathrm{E}_{\theta}\left[\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)^{2}\right]+2\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right) \mathrm{E}_{\theta}\left[\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right]+\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)^{2} \quad \mathrm{E}_{\theta}[\hat{\theta}]-\theta=\text { const. } \\
&=\mathrm{E}_{\theta}\left[\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)^{2}\right]+2\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)\left(\mathrm{E}_{\theta}[\hat{\theta}]-\mathrm{E}_{\theta}[\hat{\theta}]\right)+\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)^{2} \quad \mathrm{E}_{\theta}[\hat{\theta}]=\text { const. } \\
&=\mathrm{E}_{\theta}\left[\left(\hat{\theta}-\mathrm{E}_{\theta}[\hat{\theta}]\right)^{2}\right]+\left(\mathrm{E}_{\theta}[\hat{\theta}]-\theta\right)^{2} \\
&=\operatorname{Var}_{\theta}(\hat{\theta})+\operatorname{Bias}_{\theta}(\hat{\theta}, \theta)^{2}
\end{aligned}
$$






[最小方差无偏估计](https://en.wikipedia.org/wiki/Minimum-variance_unbiased_estimator)

# 贝叶斯估计

 ![photo]({{site.url}}/assets/img/estimate2.png)

 ![photo]({{site.url}}/assets/img/eg5-2-1.png)



# 似然

![photo]({{site.url}}/assets/img/estimate3.png)

![photo]({{site.url}}/assets/img/estimate4.png)

y=kA+w

w服从N（0，5）,A服从N（0，1），观测次数为N，估计A的值。

似然函数
$$
p(\boldsymbol{y} \mid A)=\left(\frac{1}{2 \pi \sigma_{w}^{2}}\right)^{N / 2} \exp \left[-\sum_{i=1}^{N} \frac{\left(y_{i}-A\right)^{2}}{2 \sigma_{w}^{2}}\right]
$$

对似然函数两边取对数，并对A 求偏导，结果为0

$$
\frac{\partial \ln p(\boldsymbol{y} \mid A)}{\partial A}=\frac{1}{\sigma_{n}^{2}} \sum_{i=1}^{N}\left(y_{i}-A\right)
$$

$$
=\left.\frac{N}{\sigma_{n}^{2}}\left(\frac{1}{N} \sum_{i=1}^{N} y_{i}-A\right)\right|_{A=\hat{A}_{\mathrm{m} 1}}=0
$$

解得

$$
\hat{A}_{\mathrm{ml}}=\frac{1}{N} \sum_{i=1}^{N} y_{i}
$$
均方误差
$$
\mathrm{E}\left[\left(A-\hat{A}_{\mathrm{ml}}\right)^{2}\right]=\mathrm{E}\left[\left(A-\frac{1}{N} \sum_{i=1}^{N} y_{i}\right)^{2}\right]=\frac{1}{N} \sigma_{n}^{2}
$$

# Cramer-Rao Lower bound

y=kA+w

w服从N（0，5）,A服从N（0，1），观测次数为N，估计A的值。

似然函数
$$
p(\boldsymbol{y} \mid A)=\left(\frac{1}{2 \pi \sigma_{w}^{2}}\right)^{N / 2} \exp \left[-\sum_{i=1}^{N} \frac{\left(y_{i}-A\right)^{2}}{2 \sigma_{w}^{2}}\right]
$$

对似然函数两边取对数，并对A 求二阶导

$$
\frac{\partial \ln p(\boldsymbol{y} \mid A)}{\partial A}=\frac{1}{\sigma_{w}^{2}} \sum_{i=1}^{N}\left(y_{i}-A\right)
$$

$$
\frac{\partial \ln ^2 p(\boldsymbol{y} \mid A)}{\partial A^2}=-\frac{N}{\sigma_{w}^{2}}
$$

取数学期望
$$
E[-\frac{N}{\sigma_{w}^{2}}]=-\frac{N}{\sigma_{w}^{2}}
$$
Cramer-Rao Lower bound 
$$
var(\widehat{A})\geq \frac{\sigma ^2 _w}{N}
$$

N=1时
$$
y[0]=A+w[0]
$$

$$
\widehat{A}=y[0],var(\widehat{A})=\sigma _w^2
$$

Cramer-Rao Lower bound为
$$
var(\widehat{A})\geq \sigma ^2 _w
$$


# 最小二乘

$$
y[i]=kA+w[i]
$$

观测值$y[i]$

残差平方和
$$
\sum_{n=0}^{N-1}(y[i]-kA-w[i])^2=J(A)
$$
对A求偏导全为0
$$
-2\sum _{n=0}^{N-1}(y[i]-kA-w[i])=0
$$

$$
\sum_{n=0}^{N-1}y[i]-NkA-\sum_{n=0}^{N-1}w[i]=0
$$

$$
\frac{\sum_{n=0}^{N-1}y[i]-\sum_{n=0}^{N-1}w[i]}{N}=kA
$$

$$
y_{均值}-w_{均值}=kA
$$

因为w的均值为0

所以
$$
y_{均值}=kA
$$



# 矢量估计

![photo]({{site.url}}/assets/img/estimate5.png)

# LMMSE

假设信道h的每个元素都服从[+1,-1]的均匀分布，而接收方在收到y后，仍然认为h的每个元素都服从N~(0,1)高斯分布，用LMMSE去估计。
$$
\mathbf{y}=\mathbf{X}\mathbf{h}+\mathbf{w}，假设\widehat{h}=A\mathbf{y}
$$

$$
Bmse=E\{|\mathbf{h}-\mathbf{\widehat{h}}|^2\}
=E\{(\mathbf{h}-A(\mathbf{Xh+w}))(\mathbf{h}-A(\mathbf{Xh+w}))^H\}
$$

真正的h
$$
E(h_{real})=0,D(h_{real})=\frac{1}{3}
$$
接收方认为的h
$$
E(h_{rec})=0,D(h_{rec})=1
$$
高斯分布的LMMSE估计
$$
\widehat{\boldsymbol{h}}=\boldsymbol{R}_{\boldsymbol{h}} \boldsymbol{S}^{\boldsymbol{H}}\left(\boldsymbol{S} \boldsymbol{R}_{\boldsymbol{h}} \boldsymbol{S}^{\boldsymbol{H}}+\sigma_{w}^{2} \boldsymbol{I}\right)^{-1} \boldsymbol{y}_{real}
$$

$$
\boldsymbol{R}_{\boldsymbol{h}}=diag\{1,1,1\}
$$



对于均匀分布
 ![photo]({{site.url}}/assets/img/junyun.jpg)