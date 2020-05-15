交叉验证（Cross-Validation）
================

.. contents::
  :local:
  :depth: 2


动机
----------

| 针对特定数据集训练模型很容易，但是当引入新数据时该模型如何执行？
| 您如何知道要使用哪种机器学习模型？
| 交叉验证通过确保模型产生准确的结果并将这些结果与其他模型进行比较来回答这些问题。
| 交叉验证不只是常规验证，而是通过评估模型对新数据的处理方式来分析模型对自身训练数据的处理方式。
| 以下各节讨论了几种不同的交叉验证方法：


保持方法(holdout method)
--------------


| 保持交叉验证方法涉及删除训练数据的特定部分并将其用作测试数据。
| 首先针对训练集对模型进行训练，然后要求对测试集的输出进行预测。
| 这是交叉验证技术的最简单形式，如果您有大量数据或需要快速，轻松地实施验证，则此方法很有用。

.. figure:: _img/holdout.png
   :scale: 75 %
   :alt: holdout method


| 通常，保持方法涉及将数据集分为20-30％的测试数据，其余作为训练数据。
| 这些数字可能会有所不同-较大的测试数据百分比会减少模型的训练经验，从而使您的模型更容易出错，
| 而较小比例的测试数据可能会使模型对训练数据产生不必要的偏差(bias)。
| 缺乏训练或偏差会导致 我们的模型 `Underfitting/Overfitting`_ 


.. _Underfitting/Overfitting: https://machine-learning-course.readthedocs.io/en/latest/content/overview/overfitting.html

K折交叉验证(K-Fold Cross Validation)
-----------------------

| K折交叉验证可通过对数据集的k个子集重复保持方法来帮助消除模型中的这些偏差。
| 借助K折交叉验证，数据集可分为多个唯一的测试和训练数据块。
| 使用数据的每种组合执行保持方法，并对结果求平均值以找到总误差估计。

.. figure:: _img/kfold.png
   :scale: 75 %
   :alt: kfold method


| 这里的“fold”是测试数据的唯一部分。
| 例如，如果您有100个数据点并使用10折，则每折包含10个测试点。
| K折叠交叉验证很重要，因为它允许您将完整的数据集用于培训和测试。
| 当使用较小或有限的数据集评估模型时，此功能特别有用。

.. _leave-p-out--leave-one-out-cross-validation:


留P/留一交叉验证（Leave-P-Out / Leave-One-Out Cross Validation）
--------------------------------------------


| 留P交叉验证（Leave-P-Out Cross Validation：LPOCV）通过使用模型上P个测试数据点的所有可能组合来测试模型。
| 举一个简单的例子，如果您有4个数据点并使用2个测试点，则将按照以下方式训练和测试模型：

::

    1: [ T T - - ]
    2: [ T - T - ]
    3: [ T - - T ]
    4: [ - T T - ]
    5: [ - T - T ]
    6: [ - - T T ]


| 其中“ T”是测试点，“-”是训练点。
| 下面是LPOCV的另一种可视化效果：

.. figure:: _img/LPOCV.png
   :scale: 75 %
   :alt: kfold method

   Ref: http://www.ebc.cat/2017/01/31/cross-validation-strategies/


| LPOCV可以提供极其准确的错误估计，但是对于大型数据集，它可以很快变得详尽无遗。
| 可以使用数学 `组合(combination)` n C P 来计算使用LPOCV模型必须经历的测试迭代次数，其中n是我们的数据点总数。
| 例如，我们可以看到，使用10个点的数据集和3个测试点运行LPOCV，将需要10 C 3 = 120次迭代。
| 
| 因此，留一法交叉验证（Leave-One-Out Cross Validation: LOOCV）是一种常用的交叉验证方法。它只是LPOCV的子集，P为1。
| 这使我们能够以与数据点相同的步骤数评估模型。
| LOOCV也可以看作是K折交叉验证，其中折的数量等于数据点的数量。

.. figure:: _img/LOOCV.png
   :scale: 75 %
   :alt: kfold method

   Ref: http://www.ebc.cat/2017/01/31/cross-validation-strategies/


| 与K折交叉验证相似，LPOCV和LOOCV使用完整数据集训练模型。
| 当您使用小型数据集时，它们特别有用，但会导致性能折衷。

.. _combination: https://en.wikipedia.org/wiki/Combination

.. |LPOCV| image:: http://www.ebc.cat/wp-content/uploads/2017/01/leave_p_out.png
.. |LOOCV| image:: http://www.ebc.cat/wp-content/uploads/2017/01/leave_one_out.png


结论
----------

Cross-validation is a way to validate your model against new data. The
most effective forms of cross-validation involve repeatedly testing
a model against a dataset until every point or combination of points
have been used to validate a model, though this comes with performance
trade-offs. We discussed several methods of splitting a dataset for
cross-validation:
| 交叉验证是一种针对新数据验证模型的方法。
| 交叉验证的最有效形式包括针对数据集重复测试模型，直到使用每个点或点的组合来验证模型为止，尽管这需要进行性能折衷。
| 我们讨论了分割数据集以进行交叉验证的几种方法：

- 保留方法(Holdout Method): 将一部分数据拆分为测试数据
- K折法(K-Fold Method): 将数据划分为多个部分，将每个部分用作测试/训练
- 留P法（Leave-P-Out Method）: 使用一些点（P）的每种组合作为测试数据


动机
----------

| 有许多不同类型的机器学习模型，包括线性/逻辑回归，K最近邻和支持向量机
| 但是我们如何知道哪种模型最适合我们的数据集？
| 使用不适合我们的数据的模型将导致预测的准确性降低，并可能导致财务，物理或其他形式的损害。
| 个人和公司应确保对使用的任何模型进行交叉验证。


代码示例
-------------

| 所提供的代码显示了如何使用 `Scikit-Learn`_这是一种Python机器学习库），通过三种讨论的交叉验证方法拆分一组数据。

.. _Scikit-Learn: https://scikit-learn.org

| `holdout.py`_ 使用Holdout方法拆分了一组样本糖尿病数据。
| 在scikit-learn中，这是使用称为train_test_split（）的函数完成的，该函数将一组数据随机分为两部分：

.. code:: python 

    TRAIN_SPLIT = 0.7
    ...

    dataset = datasets.load_diabetes()
    ...

    x_train, x_test, y_train, y_test = train_test_split(...)


| 请注意，
| 您可以通过更改顶部的TRAIN_SPLIT值来更改用于训练的数据部分。该数字应为0到1之间的一个数字。
| 此文件的输出显示用于拆分的训练和测试点的数量。
| 查看实际的数据点可能会有所帮助-如果您想查看这些数据点，请取消注释脚本中的最后两个打印语句。

----

| `k-fold.py`_ 使用K-Fold方法拆分一组数据。
| 这是通过创建一个KFold对象完成的，该对象初始化为要使用的拆分数量。
| Scikit-learn通过调用KFold的 `split()` 方法使拆分数据变得容易：

.. code:: python

    NUM_SPLITS = 3
    data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

    kfold = KFold(n_splits=NUM_SPLITS)
    split_data = kfold.split(data)


| 它的返回值是一个训练和测试点的数组。
| 请注意，您可以通过更改脚本顶部的关联值来使用分割数。
| 该脚本不仅输出训练/测试数据，还输出一个漂亮的进度条，您可以在其中跟踪当前折叠的进度：

::

    [ T T - - - - ]
    Train: (2: [5 6]) (3: [7 8]) (4: [ 9 10]) (5: [11 12]) 
    Test:  (0: [1 2]) (1: [3 4])
    ...

----

`leave-p-out.py`_ 使用Leave-P-Out和Leave-One-Out方法拆分一组数据。
这是通过创建LeavePOut/LeaveOneOut对象来完成的，该对象使用要使用的拆分数量初始化的LPO。
与KFold相似，训练-测试数据拆分是使用split（）方法创建的：

.. code:: python

    P_VAL = 2
    data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    loocv = LeaveOneOut()
    lpocv = LeavePOut(p=P_VAL)

    split_loocv = loocv.split(data)
    split_lpocv = lpocv.split(data)


| 请注意，您可以在脚本顶部更改P值，以查看不同值的工作方式。

.. _holdout.py: https://github.com/machinelearningmindset/machine-learning-course/tree/master/code/overview/cross-validation/holdout.py
.. _k-fold.py: https://github.com/machinelearningmindset/machine-learning-course/tree/master/code/overview/cross-validation/k-fold.py
.. _leave-p-out.py: https://github.com/machinelearningmindset/machine-learning-course/tree/master/code/overview/cross-validation/leave-p-out.py

参考资料
----------

1. https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f
2. https://machinelearningmastery.com/k-fold-cross-validation/
3. https://www.quora.com/What-is-cross-validation-in-machine-learning 
#. http://www.ebc.cat/2017/01/31/cross-validation-strategies/ 


