====================
k近邻算法(k-Nearest Neighbors)
====================

.. contents::
  :local:
  :depth: 3

介绍
-------------

| 
| K最近邻居（K-Nearest Neighbors:KNN）是机器学习的基本分类器。
|  **分类器(classifier)** 需要一个已标记的数据集，然后标记新的数据点为分类(catagories)之一。
| 因此，我们正在尝试确定对象所在的类。
| 为此，我们查看与对象最接近的点（邻居），并且拥有最多邻居的类将成为我们确定对象所属的类。 
| k是与对象最近的邻居数。
| 因此，如果k = 1，则该对象所在的class是最近邻居的class。
| 让我们看一个例子。

.. figure:: _img/knn.png
   :scale: 100 %
   :alt: KNN

   Ref: https://coxdocs.org


| 在此示例中，我们尝试将红色星形分类为绿色正方形或蓝色八边形。
| 首先，如果我们看k = 3的内圆，我们可以看到有2个蓝色八边形和1个绿色正方形。
| 因此，蓝色八角形占大多数，因此红色星号将被分类为蓝色八角形。
| 现在我们看k = 5，即外圆。在这一个中，有2个蓝色八边形和3个绿色正方形。
| 然后，红星将被分类为绿色方块。

它是如何工作的？
-----------------


| 我们将研究两种不同的解决方法。两种方法是
1.蛮力法(brute force method)
2.KD树法(K-D tree method)

暴力法(Brute Force Method)
--------------------

| 
| 这是最简单的方法。
| 基本上，它只是计算从被分类对象到集合中每个点的 **欧几里得距离(Euclidean distance)** 。
| 欧几里得距离就是连接两个点的线段的长度。
| 当点的尺寸较小或点数较小时，“暴力”方法很有用。
| 随着点数的增加，该方法必须计算欧几里德距离的次数也增加，因此该方法的性能下降。
| 幸运的是，KD树方法可以更好地处理更大的数据集。

KD树方法(K-D Tree Method)
-----------------

| 该方法试图通过减少计算欧几里得距离的次数来改善运行时间。
| 该方法背后的思想是，如果我们知道两个数据点彼此靠近，并且我们计算了到其中一个的欧几里得距离，那么我们知道该距离大致接近另一个点。
| 这是KD树的外观示例。

.. figure:: _img/KNN_KDTree.jpg
   :scale: 100 %
   :alt: KNN K-d tree

   Ref: https://slideplayer.com/slide/3273367/


| KD树的工作方式是树中的一个节点表示并保存n维图的数据。
| 每个节点代表图中的一个框。
| 
| 首先，我们可以根据一组数据构建一个KD树，
| 然后当需要对一个点进行分类时，我们只需查看该点将落在树中的位置，
| 然后计算仅在其接近的点之间的欧几里得距离，直到我们达到k个邻居。
| 
| 如果数据集较大，建议使用此方法。
| 这是因为，如果数据集较大，则创建KD树的成本相对较低，并且随着数据变大，对点进行分类的成本是恒定的。


选择k
-----------

| 选择k通常取决于您要查看的数据集。
| 您永远不要选择k = 2，因为它极有可能不会出现多数类，因此在上面的示例中将每个中都有一个，因此我们无法对红星进行分类。
| 通常，您希望k的值较小。
| 当k趋于无穷大时，所有未识别的数据点将始终归为一类或另一类，具体取决于哪一类具有更多数据点。
| 您不希望发生这种情况，因此选择较小的k是明智的。

结论
------------

以下是一些要带走(take away)的东西：

- KNN的不同方法只会影响性能，而不会影响输出
- 当点的尺寸或点数较小时，最好使用暴力法
- 当您拥有更大的数据集时，KD树方法是最好的
- SKLearn KNN分类器具有自动方法，该方法可以根据训练的数据来决定使用哪种方法。


| 选择k的值将大大改变数据的分类方式。
| 较高的k值将忽略数据的异常值（outliers），而较低的k值将赋予它们更大的权重。
| 如果k值太高，将无法对数据进行分类，因此k需要相对较小。

动机
------------


| 那么为什么有人会在另一个分类器上使用这个分类器呢？这是最好的分类器吗？这些问题的答案取决于它。
| 没有最好的分类器，这完全取决于给出分类器的数据。
| 对于一个数据集，KNN可能是最好的，但对于另一个数据集，KNN可能不是最好的。
| 最好了解其他分类器（如支持向量机 `Support Vector Machines`_），然后确定哪个分类器最好地分类了给定的数据集。

代码示例
-------------

.. code-block:: python

            # All the libraries we need for KNN
            import numpy as np
            import matplotlib.pyplot as plt

            from sklearn.neighbors import KNeighborsClassifier
            # This is used for our dataset
            from sklearn.datasets import load_breast_cancer


            # =============================================================================
            # We are using sklearn datasets to create the set of data points about breast cancer
            # Data is the set data points
            # target is the classification of those data points. 
            # More information can be found at:
            #https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
            # =============================================================================
            dataCancer = load_breast_cancer()

            # The data[:, x:n] gets two features for the data given. 
            # The : part gets all the rows in the matrix. And 0:2 gets the first 2 columns 
            # If you want to get a different two features you can replace 0:2 with 1:3, 2:4,... 28:30, 
            # there are 30 features in the set so it can only go up to 30.
            # If we wanted to plot a 3 dimensional plot then the difference between x and n needs to be 3 instead of two
            data = dataCancer.data[:, 0:2]
            target = dataCancer.target

            # =============================================================================
            # This creates the KNN classifier and specifies the algorithm being used and the k
            # nearest neighbors used for the algorithm. more information can about KNeighborsClassifier
            # can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
            #
            # Then it trains the model using the breast cancer dataset. 
            # =============================================================================
            model = KNeighborsClassifier(n_neighbors = 9, algorithm = 'auto')
            model.fit(data, target)


            # plots the points 
            plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.prism)

            # Creates the axis bounds for the grid
            axis = plt.gca()
            x_limit = axis.get_xlim()
            y_limit = axis.get_ylim()

            # Creates a grid to evaluate model
            x = np.linspace(x_limit[0], x_limit[1])
            y = np.linspace(y_limit[0], y_limit[1])
            X, Y = np.meshgrid(x, y)
            xy = np.c_[X.ravel(), Y.ravel()]

            # Creates the line that will separate the data
            boundary = model.predict(xy)
            boundary = boundary.reshape(X.shape)


            # Plot the decision boundary
            axis.contour(X, Y,  boundary, colors = 'k')

            # Shows the graph
            plt.show()


| 查看我们的代码 `knn.py`_ 以了解如何使用Python的Scikit-learn库实现ak最近邻居分类器。
| 可以在此处( `here`_)找到有关Scikit-Learn的更多信息。


| `knn.py`_, 对从Scikit-Learn的数据集库加载的一组乳腺癌数据进行分类。
| 该程序将获取数据并将其绘制在图形上，然后使用KNN算法来最好地分离数据。
| 输出应如下所示：


.. figure:: _img/knn_output_k9.png
   :scale: 100%
   :alt: KNN k = 9 output


| 绿点被分类为良性。
| 红点归类为恶性。
| 边界线是分类器做出的预测。
| 该边界线由k值确定，在这种情况下，k = 9。
| 
| 这将从Scikit-Learn的数据集库中加载数据。
| 您可以将数据更改为所需的任何数据。
| 只要确保您有数据点和一系列目标即可对这些数据点进行分类。

.. code:: python

    dataCancer = load_breast_cancer()
    data = dataCancer.data[:, :2]
    target = dataCancer.target


| 您还可以更改将更改算法的k值或n_neighbors值。
| 建议您选择较小的k。
| 
| 您也可以更改使用的算法，选项为{'auto'，'ball_tree'，'kd_tree'，'brute'}。
| 这些不会更改预测的输出，它们只会更改预测数据所需的时间。
| 
| 尝试在下面的代码中将n_neighbors的值更改为1。

.. code:: python

    model = KNeighborsClassifier(n_neighbors = 9, algorithm = 'auto')
    model.fit(data, target)

| 
| 如果将n_neighbors的值更改为1，则将按最接近该点的点进行分类。
| 输出应如下所示：

.. figure:: _img/knn_output_k1.png
   :scale: 100%
   :alt: KNN k = 1 output

| 
| 将该输出与k = 9进行比较，您会发现如何对数据进行分类有很大的不同。
| 因此，如果您想忽略离群值，则需要较高的k值，否则请选择较小的k（例如1、3或5）。
| 您可以通过选择大于100的非常高的k进行试验。
| 最终，算法会将所有数据分类为1类，并且没有行可以拆分数据。

.. _here: https://scikit-learn.org

.. _knn.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/KNN/knn.py

.. _Support Vector Machines: https://machine-learning-course.readthedocs.io/en/latest/content/supervised/linear_SVM.html


参考资料
----------

1. https://medium.com/machine-learning-101/k-nearest-neighbors-classifier-1c1ff404d265
2. https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/  
3. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html 
#. https://turi.com/learn/userguide/supervised-learning/knn_classifier.html 

