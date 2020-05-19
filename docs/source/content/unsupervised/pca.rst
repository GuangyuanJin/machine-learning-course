主成分分析(Principal Component Analysis)
============================

.. contents::
  :local:
  :depth: 2

介绍
------------
| 
| 主成分分析是一种用于获取大量互连变量并选择最适合模型的技术。
| 仅关注几个变量的过程称为 **降维(dimensionality reduction)** ，有助于降低数据集的复杂性。
| 从根本上讲，主成分分析 *汇总(summarizes)* 了数据。

.. figure:: _img/pca4.png

   Ref: https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues


动机
----------

| 主成分分析对于得出具有多个变量的给定数据集的整体线性独立趋势非常有用。
| 它使您可以从可能相关或不相关的变量中提取重要的关系。
| 主成分分析的另一种应用是显示-您可以仅代表几个主成分并绘制它们，而不必表示多个不同的变量。

降维(Dimensionality Reduction)
------------------------

| 
| 降维有两种类型：
 
1.特征消除(feature elimination)
2.特征提取(feature extraction)

|
| **特征消除(Feature elimination)** 仅涉及从我们认为不必要的数据集中修剪特征。
| 消除功能的缺点是，我们会失去从删除的功能中获得的任何潜在信息。
|
| **特征提取(Feature extraction)** 通过组合现有特征来创建新变量。
| 以某些简单性或可解释性为代价，要素提取使您可以维护要素中保存的所有重要信息。
|
| 主成分分析通过创建一组称为主成分的自变量(independent variables)来处理特征提取(feature extraction)（而不是消除(elimination)）。
| 


PCA示例
-----------


| 
| 主成分分析是通过考虑所有变量并计算一组表示它们的方向(direction)和幅度对(magnitude pairs)（向量(vectors)）来进行的。
| 例如，让我们考虑下面绘制的一个小的示例数据集：
| 
.. figure:: _img/pca1.png

   Ref: https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c

| 
| 在这里我们可以看到两个方向对，分别由红线和绿线表示。
| 在这种情况下，红点的幅度更大，因为与绿色方向相比，这些点在更大的距离上更聚集。
| 主成分分析将使用幅值(magnitude)较大的向量将数据转换为较小的特征空间，从而降低维数。
| 例如，上面的图将转换为以下内容：
| 

.. figure:: _img/pca2.png

   Ref: https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c


| 
| 通过以这种方式转换数据，我们已经忽略了对我们的模型不那么重要的功能-
| 也就是说，沿绿色维度的较大变化对我们的结果的影响将大于沿红色维度的变化。
| 
| 为简洁起见，本次讨论省略了主成分分析的数学原理，但是如果您有兴趣学习它们，我们强烈建议您访问本页底部列出的参考资料。
| 

成分数
--------------------

| 
| 在上面的示例中，我们采用了二维特征空间并将其缩减为一维。
| 但是，在大多数情况下，您将使用两个以上的变量。
| 主成分分析可用于仅删除单个功能，但减少多个功能通常很有用。
| 您可以采用几种策略来决定要执行多少个功能约简：
| 

1. **任意地（Arbitrarily）**

|    这仅涉及选择许多功能(features)以保留给定模型。
|    此方法高度依赖于您的数据集和要传达的内容。
|    例如，将您的高阶数据表示在2D空间进行可视化可能会有所帮助。
|    在这种情况下，您将执行功能简化，直到拥有两个功能。
| 

2. **累积差异百分比（Percent of cumulative variability）**

| 
|    主成分分析计算的一部分涉及寻找方差的比例，该方差在执行的每轮PCA中都接近1。
|    选择特征减少步骤数的这种方法涉及选择目标方差百分比。
|    例如，让我们看一下理论数据集在PCA各个级别上的累积方差图：
| 


.. figure:: _img/pca3.png

      Ref: https://www.centerspace.net/clustering-analysis-part-i-principal-component-analysis-pca

|    
|    上面的图像称为scree plot，它表示每个主成分的累积和当前方差比例。
|    如果我们希望至少有80％的累积方差，我们将根据此scree图使用至少6个主成分。
|    通常不建议针对100％的差异，因为达到这意味着您的数据集具有冗余数据。
|    

3. **个体差异百分比(Percent of individual variability)**


|    
|    在达到差异的累积百分比之前，不使用主要成分，而可以使用主要成分，直到新成分不会增加太多可变性为止。
|    在上图中，我们可能选择使用3个主要成分，因为下一个成分的变异性没有那么大的下降。
|    

结论
----------

| 
| 主成分分析是一种汇总数据的技术，并且根据您的用例具有很高的灵活性。
| 在显示和分析大量可能的因变量方面，它可能很有价值。
| 执行主成分分析的技术范围从任意选择主成分到自动找到它们直到达到差异为止。
| 

代码示例
------------

.. code-block:: python

            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import numpy as np

            # A value we picked to always display the same results
            # Feel free to change this to any value greater than 0 view different random value outcomes
            seed = 9000

            # We're using a seeded random state so we always get the same outcome
            seeded_state = np.random.RandomState(seed=seed)

            # Returns a random 150 points (x, y pairs) in a gaussian distribution,
            # IE most of the points fall close to the average with a few outliers
            rand_points = seeded_state.randn(150, 2)

            # The @ operator performs matrix multiplication, and serves to bring
            # our gaussian distribution points closer together
            points = rand_points @ seeded_state.rand(2, 2)
            x = points[:, 0]
            y = points[:, 1]

            # Now we have a sample dataset of 150 points to perform PCA on, so
            # go ahead and display this in a plot.
            plt.scatter(x, y, alpha=0.5)
            plt.title("Sample Dataset")

            print("Plotting our created dataset...\n")
            print("Points:")
            for p in points[:10, :]:
                print("({:7.4f}, {:7.4f})".format(p[0], p[1]))
            print("...\n")

            plt.show()

            # Find two principal components from our given dataset
            pca = PCA(n_components = 2)
            pca.fit(points)

            # Once we are fitted, we have access to inner mean_, components_, and explained_variance_ variables
            # Use these to add some arrows to our plot
            plt.scatter(x, y, alpha=0.5)
            plt.title("Sample Dataset with Principal Component Lines")
            for var, component in zip(pca.explained_variance_, pca.components_):
                plt.annotate(
                    "",
                    component * np.sqrt(var) * 2 + pca.mean_,
                    pca.mean_,
                    arrowprops = {
                        "arrowstyle": "->",
                        "linewidth": 2
                    }
                )

            print("Plotting our calculated principal components...\n")

            plt.show()

            # Reduce the dimensionality of our data using a PCA transformation
            pca = PCA(n_components = 1)
            transformed_points = pca.fit_transform(points)

            # Note that all the inverse transformation does is transforms the data to its original space.
            # In practice, this is unnecessary. For this example, all data would be along the x axis.
            # We use it here for visualization purposes
            inverse = pca.inverse_transform(transformed_points)
            t_x = inverse[:, 0]
            t_y = inverse[:, 0]

            # Plot the original and transformed data sets
            plt.scatter(x, y, alpha=0.3)
            plt.scatter(t_x, t_y, alpha=0.7)
            plt.title("Sample Dataset (Blue) and Transformed Dataset (Orange)")

            print("Plotting our dataset with a dimensionality reduction...")

            plt.show()

| 
| 我们的示例代码 `pca.py`_, 向您展示了如何对随机x，y对的数据集执行主成分分析。
| 该脚本经过很短的生成该数据的过程，然后调用sklearn的PCA模块：
| 

.. _pca.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/unsupervised/PCA/pca.py

.. code:: python

   # Find two principal components from our given dataset
   pca = PCA(n_components = 2)
   pca.fit(points)


| 
| 该过程的每个步骤都包含使用matplotlib的有用可视化。
| 例如，上面拟合的主成分被绘制为数据集上的两个向量：
| 

.. figure:: _img/pca5.png


| 
| 该脚本还显示了如何执行上述降维。
| 在sklearn中，这是通过在安装PCA之后简单地调用transform方法来完成的，或者使用fit_transform同时执行两个步骤：
| 

.. code:: python

   # Reduce the dimensionality of our data using a PCA transformation
   pca = PCA(n_components = 1)
   transformed_points = pca.fit_transform(points)


| 
| 我们的转换的最终结果只是一系列X值，尽管该代码示例执行了逆向转换以在下图中绘制结果：
| 

.. figure:: _img/pca6.png

参考资料
----------

1. http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
2. https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c
3. https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
4. https://en.wikipedia.org/wiki/Principal_component_analysis
5. https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
6. https://www.centerspace.net/clustering-analysis-part-i-principal-component-analysis-pca
