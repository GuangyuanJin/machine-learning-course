#################
线性回归
#################

.. contents::
  :local:
  :depth: 3


**********
Motivation
**********
当我们看到一个数据集时，我们尝试找出它的含义。我们在数据点之间寻找连接，看看是否可以找到任何模式。
有时很难看到这些模式，因此我们使用代码来帮助我们找到它们。
数据可以遵循许多不同的模式，因此如果我们可以缩小选择范围并减少编写代码来分析它们的话，将很有帮助。
这些模式(patterns)之一是线性关系。如果我们可以在数据中找到这种模式，则可以使用线性回归技术对其进行分析。


********
总览
********
**线性回归**是一种用于分析**输入**变量和单个**输出**变量 之间的**线性关系**的技术。
**线性关系**指的是数据点趋向于遵循一条直线。 
**简单线性回归**仅涉及单个输入变量。图1 显示了具有线性关系的数据集。

.. figure:: _img/LR.png
   
   **图1.具有线性关系的示例数据集** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression.py
   
  
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a data set for analysis

x, y = make_regression(n_samples=500, n_features = 1, noise=25, random_state=0)

# Split the data set into testing and training data

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Plot the data

sns.set_style("darkgrid")
sns.regplot(x_test, y_test, fit_reg=False)

# Remove ticks from the plot

plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
```

我们的目标是找到最能模拟数据点路径的线，称为最佳拟合线。
方程式1中的方程式是线性方程式的示例。

.. figure:: _img/Linear_Equation.png
   
   **方程1.线性方程**

*图2*显示了我们在图1中使用的数据集，其中最适合它。

.. figure:: _img/LR_LOBF.png
   
   **Figure 2. The data set from Figure 1 with a line of best fit** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression_lobf.py

让我们分解一下。我们已经知道x是输入值，y是我们的预测输出。
a₀和a₁描述了我们线的形状。a₀称为 **偏差(bias)**，a₁称为**权重(weight)**。
更改a₀将在绘图上向上或向下移动线，更改a₁会更改线的斜率。
线性回归有助于我们为a₀和a₁选取合适的值。

注意，我们可以有多个输入变量。
在这种情况下，我们称其为 **多元线性回归**。
添加额外的输入变量仅意味着我们需要找到更多权重。
对于本练习，我们将仅考虑简单的线性回归。


***********
何时使用
***********
线性回归是一种有用的技术，但并不总是适合您的数据的正确选择。
当您的自变量和因变量之间存在线性关系并且您试图预测连续值时，线性回归是一个不错的选择[ 图1 ]。

当自变量和因变量之间的关系更复杂或输出是离散值时，这不是一个好选择。
例如，图3显示的数据集没有线性关系，因此线性回归将不是一个好选择。

.. figure:: _img/Not_Linear.png
   
   **图3。没有线性关系的样本数据集** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/not_linear_regression.py

值得注意的是，有时您可以对数据应用转换，使其看起来是线性的。
例如，您可以将对数应用于指数数据以使其平坦化。
然后，您可以对转换后的数据使用线性回归。
在转换数据的一种方法sklearn是记录 here_.

.. _here: https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html

*图4*是一个看起来不是线性但可以转换为线性关系的数据示例。

.. figure:: _img/Exponential.png
   
   **图4.遵循指数曲线的示例数据集** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/exponential_regression.py

*图5*是对数转换输出变量后的相同数据。

.. figure:: _img/Exponential_Transformed.png
   
   **图5.将对数应用到输出变量后的图4的数据集** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/exponential_regression_transformed.py


*************
成本函数(Cost Function)
*************
有了预测后，我们需要某种方法来判断它是否合理。
一个 **成本函数**可以帮助我们做到这一点。
成本函数将所有预测与它们的实际值进行比较，并为我们提供一个可用来对预测函数评分的单一数字。
*图6*显示了一种这样的预测的成本。

.. figure:: _img/Cost.png
   
   **图6.图2中的图，其中强调了一个预测的代价** [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression_cost.py

成本函数中出现的两个常见术语是**误差(error)**和 **平方误差(squared error)**。
误差[ 公式2 ]是我们的预测与实际值相差多远。

.. figure:: _img/Error_Function.png
   
   **公式2. 误差函数示例**

对这个值进行平方运算，可以得出*等式3*中所示的一般误差距离(general error distance)的有用表达式。

.. figure:: _img/Square_Error_Function.png
   
   **公式3.平方误差函数的示例**

我们知道，实际值之上的2误差和实际值之下2的误差应该彼此一样严重。
平方误差使这一点很清楚，因为这两个值都导致平方误差为4。

我们将使用公式4中所示的均方误差（MSE）函数作为我们的成本函数。
此函数查找我们所有数据点的平均平方误差值。

.. figure:: _img/MSE_Function.png
   
   **公式4：均方误差（MSE）函数**

成本函数对我们很重要，因为它们可以衡量我们的模型相对于目标值的准确性。
在以后的模块中，确保模型的准确性仍然是关键主题。


*******
方法
*******
成本较低的函数意味着数据点之间的平均误差较低。
换句话说，较低的成本意味着数据集的模型更准确。
我们将简要介绍一些使成本函数最小化的方法

普通最小二乘(Ordinary Least Squares)
======================
**普通最小二乘法** 是使成本函数最小化的常用方法。
在这种方法中，我们将数据视为一个大矩阵，然后使用线性代数来估计线性方程式中系数的最佳值。
幸运的是，您不必担心做任何线性代数，因为Python代码会为您处理它。
这也恰好是用于此模块代码的方法。

以下是此模块中与普通最小二乘法有关的Python代码的相关行。

.. code-block:: python

   # 创建一个线性回归对象
   regr = linear_model.LinearRegression()

梯度下降(Gradient Descent)
================
**梯度下降法**是一种猜测线性方程式系数的迭代方法，以最小化成本函数。
该名称来自微积分中的渐变概念。
基本上，此方法将稍微移动系数的值并监视成本是否降低。
如果成本在多次迭代中持续增加，我们会停止，因为我们可能已经达到了最低要求。
可以选择停止前的迭代次数和公差来微调该方法。

以下是此模块经过修改以使用梯度下降的Python代码的相关行。

.. code-block:: python

   # 创建一个线性回归对象
   regr = linear_model.SGDRegressor(max_iter=10000, tol=0.001)


****
代码
****
该模块的主要代码位于 linear_regression_lobf.py_ 文件中。

.. _linear_regression_lobf.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression_lobf.py

该模块中的所有图形都是通过对 linear_regression.py_ 代码进行简单的修改而创建的 。

.. _linear_regression.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/linear_regression/linear_regression.py

在代码中，我们分析具有线性关系的数据集。
我们将数据分为训练集以训练我们的模型和测试集以测试其准确性。
您可能已经猜到所使用的模型基于线性回归。
我们还将显示一条最佳拟合的数据图。


**********
结论
**********
在本模块中，我们学习了线性回归。此技术可帮助我们对具有线性关系的数据进行建模。
线性关系非常简单，但是仍然会出现在许多数据集中，因此这是一个很好的技术。
学习线性回归是学习更复杂的分析技术的良好第一步。
在以后的模块中，我们将基于此处介绍的许多概念。


************
参考资料
************

1. https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
2. https://machinelearningmastery.com/linear-regression-for-machine-learning/
3. https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
#. https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
#. https://medium.com/analytics-vidhya/linear-regression-in-python-from-scratch-24db98184276
#. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#. https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html


