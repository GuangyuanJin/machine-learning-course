##############
正则化（Regularization）
##############

.. contents::
  :local:
  :depth: 3


**********
动机
**********

| 请考虑以下情形。
| 您正在制作花生酱三明治，并试图调整成分，使其具有最佳的味道。
| 您可以在决策过程中考虑面包的类型，花生酱的类型或花生酱与面包的比例。
| 但是您会考虑其他因素，例如房间的温度，早餐吃的东西或穿的袜子是什么颜色？您可能不会，因为这些东西对三明治的味道影响不大。
| 对于最终使用的任何食谱，您将更多地关注前几个功能，并避免过多地关注其他功能。
| 这是 **正则化(regularization)** 的基本思想。


********
总览
********
| 在以前的模块中，我们已经看到了在某些样本集上训练的预测模型，并根据它们与测试集的接近程度对其评分。
| 我们显然希望我们的模型做出准确的预测，但是预测是否太准确？
| 当我们查看一组数据时，有两个主要组成部分： **基本模式(underlying pattern)** 和 **噪声(noise)** 。
| 我们只想匹配模式而不是噪声。
| 考虑下面的代表二次数据的图。
| 图1使用线性模型来近似数据。
| 图2使用二次模型来近似数据。
| 图3使用高级多项式模型来近似数据。

.. figure:: _img/Regularization_Linear.png

   **Figure 1. 线性预测模型(linear prediction model)** [`code`__]

   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_linear.py

.. code-block:: python

                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.datasets import make_regression
                from sklearn.linear_model import LinearRegression
                from sklearn.pipeline import Pipeline
                import numpy as np

                # Create a data set for analysis
                x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
                y = y ** 2

                # Pipeline lets us set the steps for our modeling
                # We are using a simple linear model here
                model = Pipeline([('poly', PolynomialFeatures(degree=1)), \
                ('linear', LinearRegression(fit_intercept=False))])

                # Now we train on our data
                model = model.fit(x, y)
                # Now we pridict
                y_predictions = model.predict(x)

                # Plot data
                sns.set_style("darkgrid")
                plt.plot(x, y_predictions, color='black')
                plt.scatter(x, y, marker='o')
                plt.xticks(())
                plt.yticks(())
                plt.tight_layout()
                plt.show()


.. figure:: _img/Regularization_Quadratic.png

   **Figure 2. 二次预测模型(quadratic prediction model)** [`code`__]

   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_quadratic.py

.. code-block:: python

                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.datasets import make_regression
                from sklearn.linear_model import LinearRegression
                from sklearn.pipeline import Pipeline
                import numpy as np

                # Create a data set for analysis
                x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
                y = y ** 2

                # Pipeline lets us set the steps for our modeling
                # We are using a quadratic model here (polynomial with degree 2)
                model = Pipeline([('poly', PolynomialFeatures(degree=2)), \
                ('linear', LinearRegression(fit_intercept=False))])

                # Now we train on our data
                model = model.fit(x, y)
                # Now we pridict
                # The next two lines are used to model input for our prediction graph
                x_plot = np.linspace(min(x)[0], max(x)[0], 100)
                x_plot = x_plot[:, np.newaxis]
                y_predictions = model.predict(x_plot)

                # Plot data
                sns.set_style("darkgrid")
                plt.plot(x_plot, y_predictions, color='black')
                plt.scatter(x, y, marker='o')
                plt.xticks(())
                plt.yticks(())
                plt.tight_layout()
                plt.show()


.. figure:: _img/Regularization_Polynomial.png

   **Figure 3. 高阶多项式预测模型(high degree polynomial prediction model)** [`code`__]

   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_polynomial.py

.. code-block:: python

                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.datasets import make_regression
                from sklearn.linear_model import LinearRegression
                from sklearn.pipeline import Pipeline
                import numpy as np

                # Create a data set for analysis
                x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
                y = y ** 2

                # Pipeline lets us set the steps for our modeling
                # We are using a polynomial model here (polynomial with degree 10)
                model = Pipeline([('poly', PolynomialFeatures(degree=10)), \
                ('linear', LinearRegression(fit_intercept=False))])

                # Now we train on our data
                model = model.fit(x, y)
                # Now we pridict
                # The next two lines are used to model input for our prediction graph
                x_plot = np.linspace(min(x)[0], max(x)[0], 100)
                x_plot = x_plot[:, np.newaxis]
                y_predictions = model.predict(x_plot)

                # Plot data
                sns.set_style("darkgrid")
                plt.plot(x_plot, y_predictions, color='black')
                plt.scatter(x, y, marker='o')
                plt.xticks(())
                plt.yticks(())
                plt.tight_layout()
                plt.show()

| 图1 拟合数据不足，
| 图2 看起来很适合数据，
| 图3 拟合得非常好。
| 在上述所有模型中，第三个可能是测试集最准确的模型。但这不一定是一件好事。
| 如果再添加一些测试点，我们可能会发现第三个模型在预测它们时不再像现在那样准确，但是第二个模型仍然相当不错。这是因为第三个模型过度拟合。
| 过度拟合意味着它在拟合测试数据（包括噪声）方面确实做得非常好，但是在推广到新数据时却表现不佳。
| 第二种模型非常适合数据，并且没有那么复杂以至于不能泛化。
| 
| 正则化的目的是通过惩罚更复杂的模型来避免过度拟合。正则化的一般形式涉及在成本函数中增加一个额外项。
| 因此，如果我们使用成本函数(cost function)CF，则正则化可能导致我们将其更改为CF +λ* R，其中R是权重的某些函数，而λ是调整参数(tuning parameter)。
| 结果是权重较高（更复杂）的模型将受到更多惩罚。
| 调整参数基本上使我们可以调整正则化以获得更好的结果。
| λ越高，权重对总成本的影响越小。


*******
方法
*******
| 我们可以使用许多方法进行正则化。
| 在下面，我们将介绍一些较常见的以及何时使用它们。

岭回归（Ridge Regression）
================
|  **岭回归** 是一种正则化类型，其中函数R涉及求和权重的平方。
| 公式1显示了修改后的成本函数的示例。

.. figure:: _img/latex-ridge-eq.gif

   **Equation 1. 岭回归的成本函数**


| 公式1是正则化的示例，其中w表示我们的权重。
| 岭回归迫使权重接近零，但绝不会使权重为零。
| 这意味着所有功能都将在我们的模型中表示，但过拟合将被最小化。
| 当我们没有大量特征并且只想避免过度拟合时，岭回归是一个不错的选择。
| 图4比较了使用和不使用岭回归的模型。

.. figure:: _img/Regularization_Ridge.png

   **Figure 4. 将岭回归应用于模型** [`code`__]

   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_ridge.py

.. code-block:: python

                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.datasets import make_regression
                from sklearn.linear_model import LinearRegression, Ridge
                from sklearn.pipeline import Pipeline
                import numpy as np

                # Create a data set for analysis
                x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
                y = y ** 2

                # Pipeline lets us set the steps for our modeling
                # We are comparing a standard polynomial model against one with ridge
                model = Pipeline([('poly', PolynomialFeatures(degree=10)), \
                ('linear', LinearRegression(fit_intercept=False))])
                regModel = Pipeline([('poly', PolynomialFeatures(degree=10)), \
                ('ridge', Ridge(alpha=5.0))])

                # Now we train on our data
                model = model.fit(x, y)
                regModel = regModel.fit(x, y)
                # Now we pridict
                # The next four lines are used to model input for our prediction graph
                x_plot = np.linspace(min(x)[0], max(x)[0], 100)
                x_plot = x_plot[:, np.newaxis]
                y_plot = model.predict(x_plot)
                yReg_plot = regModel.predict(x_plot)

                # Plot data
                sns.set_style("darkgrid")
                plt.plot(x_plot, y_plot, color='black')
                plt.plot(x_plot, yReg_plot, color='red')
                plt.scatter(x, y, marker='o')
                plt.xticks(())
                plt.yticks(())
                plt.tight_layout()
                plt.show()

| 在图4中，黑线表示未应用Ridge回归的模型，红线表示已应用Ridge回归的模型。
| 请注意红线的平滑程度。针对将来的数据，它可能会做得更好。

在 regularization_ridge.py_ 文件中, 添加岭回归的代码为：

.. _regularization_ridge.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_ridge.py

.. code-block:: python

    regModel = Pipeline([('poly', PolynomialFeatures(degree=6)), ('ridge', Ridge(alpha=5.0))])

| 添加Ridge回归就像在Pipeline调用中添加一个附加参数一样简单。
| 在这里，参数alpha表示我们的调整变量。
| 有关scikit-learn中Ridge回归的更多信息，请参见`here`__.

.. __: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

Lasso Regression
================

|  **Lasso regression** 是一种正则化类型，其中函数R涉及求和权重的绝对值。
| 公式2显示了修改后的成本函数的示例。

.. figure:: _img/latex-lasso-eq.gif

   **Equation 2. lasso回归的成本函数**

| 公式2是正则化的示例，其中w表示我们的权重。
| 请注意，ridge回归和lasso回归的相似程度。唯一明显的区别是 **权重的平方** 。这恰好对他们的工作产生了重大影响。
| 与ridge回归不同，lasso回归可以将权重设为零。这意味着我们生成的模型甚至可能不会考虑某些功能！
| 在我们拥有一百万个仅需少量重要功能的功能的情况下，这是非常有用的结果。
| lasso索回归使我们避免过度拟合，而将注意力集中在所有功能的一小部分上。
| 在原始情况下，我们最终将忽略那些对我们的三明治饮食体验没有太大影响的因素。
| 图5 给出了应用lasso回归和不应用lasso回归的模型的比较。

.. figure:: _img/Regularization_Lasso.png

   **Figure 5. Lasso回归应用于模型** [`code`__]

   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_lasso.py

.. code-block:: python

                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.datasets import make_regression
                from sklearn.linear_model import LinearRegression, Lasso
                from sklearn.pipeline import Pipeline
                import numpy as np

                # Create a data set for analysis
                x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
                y = y ** 2

                # Pipeline lets us set the steps for our modeling
                # We are comparing a standard polynomial model against one with lasso
                model = Pipeline([('poly', PolynomialFeatures(degree=10)), \
                ('linear', LinearRegression(fit_intercept=False))])
                regModel = Pipeline([('poly', PolynomialFeatures(degree=10)), \
                ('lasso', Lasso(alpha=5, max_iter=1000000))])

                # Now we train on our data
                model = model.fit(x, y)
                regModel = regModel.fit(x, y)
                # Now we pridict
                x_plot = np.linspace(min(x)[0], max(x)[0], 100)
                x_plot = x_plot[:, np.newaxis]
                y_plot = model.predict(x_plot)
                yReg_plot = regModel.predict(x_plot)

                # Plot data
                sns.set_style("darkgrid")
                plt.plot(x_plot, y_plot, color='black')
                plt.plot(x_plot, yReg_plot, color='red')
                plt.scatter(x, y, marker='o')
                plt.xticks(())
                plt.yticks(())
                plt.tight_layout()
                plt.show()


| 在上图中，黑线表示未应用Lasso回归的模型，红线表示已应用Lasso回归的模型。红线比黑线平滑得多。
| 将Lasso回归应用于10阶模型，但结果看起来它的阶数要低得多！
| Lasso模型可能会更好地处理未来的数据。

 regularization_lasso.py_ 文件中，添加Lasso回归的代码是：


.. _regularization_lasso.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/overview/regularization/regularization_lasso.py

.. code-block:: python

  regModel = Pipeline([('poly', PolynomialFeatures(degree=6)), \
  ('lasso', Lasso(alpha=0.1, max_iter=100000))])


| 添加Lasso回归与添加Ridge回归一样简单。
| 在这里，参数alpha表示我们的调整变量，并max_iter表示要运行的最大迭代次数。
| 有关scikit-learn中Lasso回归的更多信息，请参见 `here`__.

.. __: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

*******
摘要
*******
In this module, we learned about regularization. With regularization, we have
found a good way to avoid overfitting our data. This is a common but important
problem in modeling so it's good to know how to mediate it. We have also
explored some methods of regularization that we can use in different
situations. With this, we have learned enough about the core concepts of
machine learning to move onto our next major topic, supervised learning.
| 在本模块中，我们学习了正则化。通过 **正则化(regularization)** ，我们找到了 **避免过拟合** 数据的好方法。
| 这是建模中一个常见但重要的问题，因此最好了解如何进行调解。
| 我们还探索了一些可以在不同情况下使用的正则化方法。
| 到此为止，我们已经对机器学习的核心概念有了足够的了解，可以进入下一个主要主题监督学习。


************
参考资料
************

1. https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a
2. https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques 
3. https://www.quora.com/What-is-regularization-in-machine-learning
#. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html 
#. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html


