---
layout: post
title: 对称矩阵
categories:
tags:
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
> <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
# 奇异值分解

定义矩阵A的SVD为$$A = U \Sigma V^{T}$$。其中$$U \in \mathbb{R}^{m \times m}$$，$$\Sigma \in \mathbb{R}^{m \times n}$$，Σ是一个对角矩阵，主对角线上的每个元素都称为奇异值，$$V \in \mathbb{R}^{n \times n}$$。U和V都是酉矩阵。
$$A^{T} A \in \mathbb{R}^{n \times n}$$进行特征分解，将$$A^{T} A$$的所有特征向量张成一个n×n的矩阵V，就是SVD公式里面的V矩阵。一般V中的每个特征向量叫做A的右奇异向量。

$$A=U \Sigma V^{T} \Rightarrow A^{T}=V \Sigma^{T} U^{T} \Rightarrow A^{T} A=V \Sigma^{T} U^{T} U \Sigma V^{T} \Rightarrow A^{T} A=V \Sigma^{2} V^{T}$$

$$A A^{T} \in \mathbb{R}^{m \times m}$$进行特征分解。将$$A A^{T}$$的所有特征向量张成一个m×m的矩阵U，就是SVD公式里面的U矩阵。一般将U中的每个特征向量叫做A的左奇异向量。

$$A A^{T}=U \Sigma V^{T} V \Sigma^{T} U^{T}=U \Sigma^{2} U^{T}$$

由于Σ除了对角线上是奇异值其他位置都是0，奇异值σ
$$A=U \Sigma V^{T} \Rightarrow A V=U \Sigma V^{T} V \Rightarrow A V=U \Sigma \Rightarrow A v_{i}=u_{i} \sigma_{i} \Rightarrow \sigma_{i}=\frac{A v_{i}}{u_{i}}$$

定义矩阵A的SVD为$$A = U \Sigma V^{T}$$。其中$$U \in \mathbb{R}^{m \times m}$$，$$\Sigma \in \mathbb{R}^{m \times n}$$，Σ是一个对角矩阵，主对角线上的每个元素都称为奇异值，$$V \in \mathbb{R}^{n \times n}$$。U和V都是酉矩阵。



# 



由$$A A^{T}=U \Sigma V^{T} V \Sigma^{T} U^{T}=U \Sigma^{2} U^{T}$$知，m×n矩阵A的奇异值$$\sigma_{i}$$是矩阵乘积$$A A^{T}$$的特征值的**正**平方根。奇异值都是正数，但特征值可能有负数。

对于对称矩阵，如果所有特征值都大于等于0，则它的特征分解就是奇异值分解。

对于半正定（或正定）Hermitian矩阵，它的特征分解就是奇异值分解。



半正定矩阵：特征值大于等于零的实对称矩阵。

正定矩阵：特征值都大于零的实对称矩阵。

Hermitian矩阵是复共轭对阵矩阵，实对称矩阵是Hermitian矩阵的特例。

例：

$$mat_1=\left[ \begin{matrix} 2& 1\\1&2\end{matrix} \right]$$，是一个对称矩阵且特征值都大于等于零，因此特征分解和奇异值分解相同。

特征分解

![image-20211007203130109](C:\Users\JESSICA\AppData\Roaming\Typora\typora-user-images\image-20211007203130109.png)

奇异值分解

![image-20211007203056725](C:\Users\JESSICA\AppData\Roaming\Typora\typora-user-images\image-20211007203056725.png)

$$mat_1=\left[ \begin{matrix} 1& 0\\0&-2\end{matrix} \right]$$，虽然是对称矩阵，但特征值有负数，奇异值分解和特征分解不相同。

特征分解

![image-20211007203340234](C:\Users\JESSICA\AppData\Roaming\Typora\typora-user-images\image-20211007203340234.png)

奇异值分解

![image-20211007203328587](C:\Users\JESSICA\AppData\Roaming\Typora\typora-user-images\image-20211007203328587.png)

# 



对于矩阵

证明半正定矩阵特征值非负：

对于实对称阵A，一定存在可逆阵P，使得
$$P^TAP=diag(a_1,a_2,...,a_n)$$
其中$$a_1,a_2,...,a_n$$为A的特征值。(实对称矩阵可以正交相似对角化，P是正交矩阵，对角阵元素是A的特征值)

对于任意列向量$$Y=[y_1,y_2,...,y_n]^T$$，
做列向量$$X=PY$$。
由于A半正定，所以$$X^TAX\geq 0$$
$$[(PY)^T]A(PY)\geq 0$$
$$Y^T[(P^T)AP]Y\geq 0$$
$$a_1*y_1^2+a_2*y_2^2+...+a_n*y_n^2\geq 0$$
由于列向量Y的任意性，
所以A的特征值$$a_1,a_2,...,a_n \geq 0$$





左奇异向量实际上就是 ![[公式]](https://www.zhihu.com/equation?tex=AA%5E%7BT%7D) 的特征向量，右奇异向量是 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7BT%7DA) 的特征向量

Sheldon Axler的《线性代数应该这样学（Linear Algebra Done Right）》。

# 对称矩阵

对称矩阵$$A$$是一个其元素$$a_{ij}$$关于主对角线对称的实正方矩阵，即有$$A^T=A$$或$$a_{ij}=a_{ji}$$

# Hermitian矩阵

一个正方矩阵$$A=[a_{ij}]\in C^{m×n}$$称为Hermitian矩阵，若$$A=A^H$$，其中，$$A^H=(A^*)^T=[a_{ji}^*]$$，换言之，**Hermitian矩阵是一种复共轭对称**矩阵。

实对称矩阵是Hermitian的特例。

Hermitian矩阵A的特征值一定是实数。

任何一个Hermitian矩阵是可对角化的，即$$U^-1AU=\Sigma,A是Hermitian矩阵$$。

Hermitian矩阵的所有特征向量线性无关，并且相互正交。

一个Hermitian矩阵A的特征值都是非负的，当且仅当A是非负定（或半正定）的。

**一个Hermitian矩阵A的特征值都是正的，当且仅当A是正定的。**

# 二次型

任意一个正方矩阵A的二次型$$x^HAx$$是一个实标量。以实矩阵为例，考查二次型

$$x^HAx=[x_1,x_2,x_3]\left[ \begin{matrix} 1 & 4 & 2\\ -1 & 7 & 5\\ -1 & 6 & 3\end{matrix} \right]\left[ \begin{matrix} x_1\\ x_2\\x_3\end{matrix} \right]$$

$$=x_1^2+7x_2^2+3x_3^2+3x_1x_2+x_1x_3+11x_2x_3$$

这是变元x的二次型函数，故称$$x^HAx$$为矩阵A的二次型。

# 正定矩阵

如果将大于零的二次型$$x^HAx$$称为正定的二次型，则与之对应的Hermitian矩阵称为正定矩阵。

正定矩阵即特征值都大于零的实对称矩阵。

$一个复共轭对称矩阵A称为\left\{\begin{matrix}正定矩阵，若二次型x^HAx>0，\forall x \neq 0;\\半正定矩阵，若二次型x^HAx \geq 0，\forall x \neq 0;\\负定矩阵，若二次型x^HAx<0，\forall x \neq 0;\\半负定矩阵，若二次型x^HAx \leq 0，\forall x \neq 0;
\\不定矩阵，若二次型x^HAx既可能取正值，也可能取负值。\end{matrix}\right.$

对于实观测数据向量$$x(t)$$，其自相关矩阵$$R=E\{x(t)x^T(t)\}$$是实对称矩阵；而复观测数据向量$$x(t)$$的自相关矩阵$$R=E\{x(t)x^T(t)\}$$是Hermitian矩阵。