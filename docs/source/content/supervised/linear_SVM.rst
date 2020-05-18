==========================
线性支持向量机(Linear Support Vector Machines)
==========================

.. contents::
  :local:
  :depth: 3

介绍
-------------

| **支持向量机** (Support Vector Machine: SVM) 是另一种机器学习算法，该算法用于分类数据。
| SVM的重点是尝试找到一条线(line)或超平面(hyperplane)来划分一个维度空间(a dimensional space)，从而最好地对数据点进行分类。
| 如果我们试图划分两个类A和B，我们将尝试用一条线将这两个类最好地分开。线/超平面的一侧将是A类的数据，而另一侧将是B类的数据。
| 这种分类法在分类中非常有用，因为我们必须一次计算最佳的线或超平面，任何新的数据点都可以轻松地计算出来。
| 仅通过查看它们落在行的哪一侧即可对其进行分类。
| 这与k最近邻居算法相反，在k算法中，我们必须计算每个数据点最近的邻居。
 
 
超平面（Hyperplane）
----------
|  **超平面** 取决于它所在的空间，但它划分空间分成两个断开部分。
|  例如，一维空间将只是一个点，二维空间将是一条线，三维空间将是一个平面，依此类推。


我们如何找到那个最佳的超平面/线？
----------------------------------------

| 您可能想知道可能会有多行将数据很好地拆分。
| 实际上，有无数行可以划分两个类。
| 正如您在下图中所看到的，每一行都将正方形和圆形分开，那么我们选择哪一个呢？

.. figure:: _img/Possible_hyperplane.png
   :scale: 100%
   :alt: Possible_Hyperplane

   Ref: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 


| 那么SVM如何找到将这两类分开的理想路线呢？它不只是随机选择一个。
| 该算法选择具有 **最大边距（maximum margin）** 的线/超平面。
| 最大化边距将为我们提供对数据进行分类的最佳行。如下图所示。

.. figure:: _img/optimal_hyperplane.png
   :scale: 100%
   :alt: Optimal_Hyperplane

   Ref: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 

如何最大化边距？
---------------------------

| 最接近直线的数据决定了最佳直线。这些数据点称为 **support vectors（支持向量）**。
| 它们显示为上方的正方形和圆形填充。这些向量到超平面的距离称为 **边距（margin）** 。
| 通常，这些点距超平面越远，正确分类数据的可能性就越大。
| 找到支持向量并最大化边距会涉及很多复杂的数学运算。我们不会去讨论；我们只想了解SVM背后的基本思想。

忽略异常值（Ignore Outliers）
---------------

| 有时数据类会有 **异常值（outliers）** 。这些数据点与其他分类显然是分开的。
| 支持向量机将忽略这些离群值。
| 如下图所示。


.. figure:: _img/SVM_Outliers.png
   :scale: 100%
   :alt: Outliers

   Ref:  https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/


带有红色圆圈的星星是异常值。因此，SVM将忽略异常值，并创建最佳行以将两个类分开。


内核SVM（Kernel SVM）
-----------

| 将存在无法用简单的线或超平面分开的数据类。
| 这被称为 **非线性可分离数据（non-linearly separable data）**。
| 这是此类数据的示例。

.. figure:: _img/SVM_Kernal.png
   :scale: 100%
   :alt: Kernel

   Ref:  https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/



| 没有明确的方法将星星与圆圈分开。
| SVM将能够使用称为 **内核技巧（kernel trick）** 的技巧对非线性可分离数据进行分类。
| 基本上，内核技巧将点指向更高的维度，以将非线性可分离数据转换为线性可分离数据。
| 因此，上图将用圆圈分隔，以分隔数据。
| 
| 这是内核技巧的一个示例。

.. figure:: _img/SVM_Kernel2.png
   :scale: 100%
   :alt: Kernel X Y graph

   Ref:  https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/


有三种类型的内核：

- **线性（Linear）** Kernel
- **多项式（Polynomial）** Kernel
- **径向基函数（Radial Basis Function：RBF）** kernel


| 您可以通过将“ model = svm.SVC（kernel ='linear'，C = 10000）”中的内核值更改为'poly'或'rbf'来查看这些内核如何更改最佳超平面的结果。
| 这是在linear_svm.py中。


结论
-----------
| 
| SVM是一种很好的机器学习技术，可以对数据进行分类。
| 现在，我们对SVM有所了解，我们可以展示使用此分类器的优缺点。


| SVM的优点：

- 有效地对高维空间进行分类
- 节省内存空间，因为它仅使用支持向量来创建最佳行。
- 数据点可分离时的最佳分类器


与SVM的缺点：

- 有大量数据集时效果不佳，训练时间更长。
- 当类重叠时（即不可分离的数据点），性能会很差。


动机
----------

| 为什么要使用SVM？可以对数据进行分类的模型很多。
| 为什么要使用这个？如果您知道数据点很容易分离，那么这可能是最好的分类器。
| 另外，可以使用内核技巧来扩展它，因此请尝试使用不同的内核，例如径向基函数（Radial Basis Function：RBF）。


代码示例
-------------

查看我们的代码, `linear_svm.py`_ 了解如何使用Python的Scikit-learn库实现线性SVM。可以在此处 `here`_ 找到有关Scikit-Learn的更多信息。

.. code:: python

            # All the libraries we need for linear SVM
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn import svm
            # This is used for our dataset
            from sklearn.datasets import load_breast_cancer


            # =============================================================================
            # We are using sklearn datasets to create the set of data points about breast cancer
            # Data is the set data points
            # target is the classification of those data points. 
            # More information can be found athttps://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
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
            # Creates the linear svm model and fits it to our data points
            # The optional parameter will be default other than these two,
            # You can find the other parameters at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            # =============================================================================
            model = svm.SVC(kernel = 'linear', C = 10000)
            model.fit(data, target)


            # plots the points 
            plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.prism)

            # Creates the axis bounds for the grid
            axis = plt.gca()
            x_limit = axis.get_xlim()
            y_limit = axis.get_ylim()

            # Creates a grid to evaluate model
            x = np.linspace(x_limit[0], x_limit[1], 50)
            y = np.linspace(y_limit[0], y_limit[1], 50)
            X, Y = np.meshgrid(x, y)
            xy = np.c_[X.ravel(), Y.ravel()]

            # Creates the decision line for the data points, use model.predict if you are classifying more than two 
            decision_line = model.decision_function(xy).reshape(Y.shape)


            # Plot the decision line and the margins
            axis.contour(X, Y,  decision_line, colors = 'k',  levels=[0], 
                       linestyles=['-'])
            # Shows the support vectors that determine the desision line
            axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                       linewidth=1, facecolors='none', edgecolors='k')

            # Shows the graph
            plt.show()

`linear_svm.py`_, 对从Scikit-Learn的数据集库加载的一组乳腺癌数据进行分类。
该程序将获取数据并将其绘制在图形上，然后使用SVM创建超平面来分离数据。
它还绕圈了确定超平面的支持向量。输出应如下所示：


.. figure:: _img/linear_svm_output.png
   :scale: 100%
   :alt: Linear SVM output

| 绿点被分类为良性（benign）。
| 红点归类为恶性（malignant）。



| 这将从Scikit-Learn的数据集库中加载数据。您可以将数据更改为所需的任何数据。
| 只需确保您拥有数据点和一系列目标即可对这些数据点进行分类。

.. code:: python

    dataCancer = load_breast_cancer()
    data = dataCancer.data[:, :2]
    target = dataCancer.target


| 您也可以将内核更改为“ rbf”或“polynomial”。
| 这将创建一个不同的超平面来对数据进行分类。
| 您可以在以下代码中对其进行更改：

.. code:: python

    model = svm.SVC(kernel = 'linear', C = 10000)
    model.fit(data, target)


.. _here: https://scikit-learn.org

.. _linear_svm.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/Linear_SVM/linear_svm.py


参考资料
----------

1. https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
2. https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
3. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
#. https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
#. https://towardsdatascience.com/support-vector-machines-svm-c9ef22815589


