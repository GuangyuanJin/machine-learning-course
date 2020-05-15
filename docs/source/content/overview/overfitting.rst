================================
过拟合(Overfitting)和欠拟合(Underfitting)
================================

.. contents::
  :local:
  :depth: 3

----------------------------
总览
----------------------------
| 使用机器学习时，有很多方法出错。机器学习中最常见的一些问题是 **过拟合** 和 **欠拟合** 。
| 为了理解这些概念，让我们想象一下一个机器学习模型，该模型试图学习对数字进行分类，并且可以访问训练数据集和测试数据集。

----------------------------
过拟合(Overfitting)
----------------------------

| 当模型从训练数据中学到太多时，就会遭受 **过拟合** 的影响，结果在实践中表现不佳。
| 这通常是由于模型过多地暴露于训练数据引起的。
| 对于数字分类示例，如果模型以这种方式过拟合，则可能是在误导的微小细节上出现，例如偏离标记表示特定数字。
| 
| 当您查看图表中间时，估算值看起来不错，但是边缘的误差很大。
| 实际上，此错误并不总是在极端情况下出现，并且可能在任何地方弹出。
| 训练中的噪音可能会导致错误，如下图所示。

.. figure:: _img/Overfit_small.png
   :scale: 100 %
   :alt: Overfit
(Created using https://www.desmos.com/calculator/dffnj2jbow)


| 在此示例中，数据因多项式阶而过拟合。
| 所示的点对函数y = x ^ 2是正确的，但在这些点之外并不能很好地近似函数。

----------------------------
欠拟合（Underfitting）
----------------------------

| 
| 如果模型没有从训练数据中学到足够的知识，就会遭受 **欠拟合** 的困扰，结果在实践中表现不佳。
| 与先前的想法形成直接对比，此问题是由于没有让模型从训练数据中学到足够的知识而引起的。
| 在数字分类示例中，如果训练集太小或模型没有足够的尝试从中学习，则训练集将无法挑选出数字的关键特征。
| 
| 该估计的问题在人眼中很明显，该模型应该是非线性的，而训练出来的只是一条简单的线。
| 在机器学习中，这可能是拟合不足的结果，该模型没有足够的培训数据来适应它，并且目前处于简单状态。

.. figure:: _img/Underfit.PNG
   :scale: 100 %
   :alt: Underfit
(Created using Wolfram Alpha)

----------------------------
动机
----------------------------

| 
| 寻找合适的拟合是机器学习中的核心问题之一。
| 甚至在担心特定方法之前都可以很好地掌握如何避免拟合问题，从而使模型步入正轨。
| 对拟合拥有合适的狩猎心态，而不是花更多的时间学习模型，是非常重要的。

----------------------------
Code
----------------------------
| 
| 过度拟合的示例代码显示了一些基于多项式插值的基本示例，试图查找图的方程。
| 在 overfitting.py_ 文件中，您可以看到正在建模一个真实的函数，以及一些估计不准确的估计。

.. _overfitting.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/overfitting/overfitting.py

| 估计值代表过拟合和欠拟合。
| 对于过度拟合，使用了更高次的多项式（x求立方而不是平方）。尽管所选点的数据相对较近，但它们之外还有一些伪像。
| 但是，欠拟合的示例在许多方面甚至都无法达到精度。欠拟合类似于在对二次函数建模时具有线性模型。
| 该模型在其训练的点上效果很好，在这种情况下，该点用于线性估计，但在其他方面效果不佳。

.. code-block:: python

            import matplotlib.pyplot as plt

            def real_funct(x):
                return [-(i**2) for i in x]

            def over_funct(x):
                return [-0.5*(i**3) - (i**2) for i in x]

            def under_funct(x):
                return [6*i + 9 for i in x]

            #create x values, and run them through each function
            x = range(-3, 4, 1)
            real_y = real_funct(x)
            over_y = over_funct(x)
            under_y = under_funct(x)

            #Use matplotlib to plot the functions so they can be visually compared.
            plt.plot(x, real_y, 'k', label='Real function')
            plt.plot(x, over_y, 'r', label='Overfit function')
            plt.plot(x, under_y, 'b', label='Underfit function')
            plt.legend()
            plt.show()

            #Output the data in a well formatted way, for the more numerically inclined.
            print("An underfit model may output something like this:")
            for i in range(0, 7):
                print("x: "+ str(x[i]) + ", real y: " + str(real_y[i]) + ", y: " + str(under_y[i]))

            print("An overfit model may look a little like this")
            for i in range(0, 7):
                print("x: "+ str(x[i]) + ", real y: " + str(real_y[i]) + ", y: " + str(over_y[i]))


----------------------------
结论
----------------------------
| 
| 查看交叉验证(cross-validation)和正则化部分(regularization sections)，以获取有关如何避免机器学习模型过度拟合的信息。
| 理想情况下，合适的外观如下所示：

.. figure:: _img/GoodFit.PNG
   :scale: 100 %
   :alt: Underfit
(Created using Wolfram Alpha)


| 
| 当以任何能力使用机器学习时，经常会出现诸如过度拟合之类的问题，并且掌握这一概念非常重要。
| 本节中的模块是整个存储库中最重要的模块之一，因为无论采用哪种实现，机器学习始终包括这些基础知识。


----------
参考资料
----------

1. https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/
2. https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76
3. https://towardsdatascience.com/overfitting-vs-underfitting-a-conceptual-explanation-d94ee20ca7f9

