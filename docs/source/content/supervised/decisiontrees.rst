决策树(Decision Trees)
==============

.. contents::
  :local:
  :depth: 2

介绍
------------

| 决策树是机器学习中的一个分类器，它使我们能够根据先前的数据进行预测。
| 它们就像一系列连续的“ if…then”语句，您将新数据馈入它们以获得结果。
| 
| 为了演示决策树，让我们看一个例子。
| 想象一下，我们要预测麦克是否会在任何一天去杂货店购物。
| 我们可以看看导致Mike前往商店的先前因素：

.. figure:: _img/shopping_table.png
   :alt: Dataset

   **Figure 1. An example dataset**


| 在这里，我们可以查看Mike的杂货供应量，天气以及Mike每天是否工作。
| 绿色的行是他去商店的日子，红色的行是他不去的日子。
| 决策树的目的是试图理解Mike 为何要去商店，然后将其应用于新数据。
| 
| 让我们将第一个属性分成一棵树。
| 迈克的补给量可能low, medium, or high：

.. figure:: _img/decision_tree_1.png
   :alt: Tree 1

   **Figure 2. Our first split**

| 
| 在这里，我们可以看到Mike的补给量很高，就永远不会去商店。
| 这称为 **纯子集(pure subset)** ，即仅包含正例(positive examples)或仅负例(negative examples)的子集。
| 使用决策树，无需进一步分解纯子集。
| 
| 让我们将Med Supplies类别划分为Mike那天是否工作：

.. figure:: _img/decision_tree_2.png
   :alt: Tree 2

   **Figure 3. Our second split**

| 在这里我们可以看到我们还有两个纯子集，因此这棵树是完整的。
| 我们可以用它们各自的答案替换任何纯子集-在这种情况下，yes或no。
| 
| 最后，让我们按“天气”属性划分“Low Supplies”类别：

.. figure:: _img/decision_tree_3.png
   :alt: Tree 3

   **Figure 4. Our third split**

| 现在我们有了所有纯子集，我们可以创建最终决策树(final decision tree)：


.. figure:: _img/decision_tree_4.png
   :alt: Tree 4

   **Figure 5. The final decision tree**

动机
----------


| 决策树易于创建，可视化和解释。
| 因此，它们通常是用于对数据集建模的第一种方法。
| 决策树的层次结构和分类性质使其实现起来非常直观。
| 决策树根据您拥有的数据点数对数展开，这意味着较大的数据集对树的创建过程的影响将小于其他分类器。
| 由于树结构，对新数据点的分类也可以对数地(logarithmically)执行。

分类和回归树(Classification and Regression Trees)
-----------------------------------

| 决策树算法也称为分类和回归树(Classification and Regression Trees: CART)。
|  **分类树(Classification Tree)** ，如上所示的一个，用来获取来自一组可能的值的结果。
| 一个 **回归树(Regression Tree)**是决策树，结果就是一个连续的值，如汽车的价格。


分裂（归纳）Splitting (Induction)
---------------------

| 决策树是通过称为 **induction** 的拆分过程创建的 ，但是我们如何知道何时拆分？
| 我们需要一种确定最佳属性的递归算法。
| 一种这样的算法是 **贪婪算法(greedy algorithm)** ：

1. 从根开始，我们为每个属性创建一个拆分。
2. 对于每个创建的拆分，请计算拆分成本。
3. 选择成本最低的拆分。
4. 递归到子树，然后从步骤1继续。

| 重复此过程，直到所有节点都具有与目标结果相同的值，或者拆分不会为预测增加任何值。
| 该算法将根节点作为最佳分类器(best classifier)。


分割成本(Cost of Splitting)
-----------------

| 拆分的成本由 **成本函数（cost function）** 确定。
| 使用成本函数的目的是以一种可以计算的方式拆分数据，并提供最大的信息收益。

|
| 对于分类树，那些提供答案而不是值的树，我们可以使用 **Gini Impurities** 计算信息增益：

.. figure:: _img/Gini_Impurity.png

    **Equation 1. The Gini Impurity Function**

    Ref: https://sebastianraschka.com/faq/docs/decision-tree-binary.html

.. figure:: _img/Gini_Information_Gain.png

    **Equation 2. The Gini Information Gain Formula**

    Ref: https://sebastianraschka.com/faq/docs/decision-tree-binary.html

| 
| 为了计算信息增益，我们首先开始计算根节点的 Gini Impurity。
| 让我们看一下我们先前使用的数据：

+-----+----------+----------+----------+----------+
|     | Supplies | Weather  | Worked?  | Shopped? |
+=====+==========+==========+==========+==========+
| D1  | Low      | Sunny    | Yes      | Yes      |
+-----+----------+----------+----------+----------+
| D2  | High     | Sunny    | Yes      | No       |
+-----+----------+----------+----------+----------+
| D3  | Med      | Cloudy   | Yes      | No       |
+-----+----------+----------+----------+----------+
| D4  | Low      | Raining  | Yes      | No       |
+-----+----------+----------+----------+----------+
| D5  | Low      | Cloudy   | No       | Yes      |
+-----+----------+----------+----------+----------+
| D6  | High     | Sunny    | No       | No       |
+-----+----------+----------+----------+----------+
| D7  | High     | Raining  | No       | No       |
+-----+----------+----------+----------+----------+
| D8  | Med      | Cloudy   | Yes      | No       |
+-----+----------+----------+----------+----------+
| D9  | Low      | Raining  | Yes      | No       |
+-----+----------+----------+----------+----------+
| D10 | Low      | Raining  | No       | Yes      |
+-----+----------+----------+----------+----------+
| D11 | Med      | Sunny    | No       | Yes      |
+-----+----------+----------+----------+----------+
| D12 | High     | Sunny    | Yes      | No       |
+-----+----------+----------+----------+----------+

| 
| 我们的根节点是目标变量，无论迈克是否打算去购物。
| 要计算其Gini Impurity，我们需要找到每个结果的概率平方和，然后从一个结果中减去该结果：
| 

.. figure:: _img/Gini_1.png

.. figure:: _img/Gini_2.png

.. figure:: _img/Gini_3.png

| 
| 如果我们在第一个属性“Supplies”上划分，让我们计算基尼信息增益（Gini Information Gain）。
| 我们可以分为三个不同的类别-Low, Med, and High。
| 对于这些，我们计算其 Gini Impurity：
| 

.. figure:: _img/Gini_4.png

.. figure:: _img/Gini_5.png

.. figure:: _img/Gini_6.png


| 如您所见，High supplies的impurity为0。
| 这意味着，如果我们分割Supplies并收到高输入，我们将立即知道结果是什么。
| 为了确定该拆分的Gini Information Gain，我们计算根的impurity减去每个孩子的impurity的加权平均值：

.. figure:: _img/Gini_7.png

.. figure:: _img/Gini_8.png


| 对于每个可能的分割，我们都会继续使用此模式，然后选择能够为我们提供最高信息增益值的分割。
| 最大化信息增益使我们可以进行最极化的分割，从而降低了新输入被错误分类的可能性。

修剪（Pruning）
-------


| 通过足够大的数据集创建的决策树最终可能会产生过多的拆分，每个拆分的有用性都会降低。
| 高度详细的决策树甚至可能导致过度拟合，如上一模块中所述。
| 因此，修剪掉决策树中次要的部分是很有益的。
| 修剪涉及计算每个结束子树（叶节点及其父节点）的信息增益，然后删除信息增益最小的子树：

.. figure:: _img/Dec_Trees_Pruning.png

    Ref: http://www.cs.cmu.edu/~bhiksha/courses/10-601/decisiontrees/


| 如您所见，子树被更突出的结果所取代，成为新的叶子。
| 可以重复此过程，直到达到所需的复杂性级别，树的高度或信息获取量（information gain amount）。
| 由于树是为了节省修剪时间而构建的，因此可以跟踪和存储信息增益。
| 每个模型都应使用自己的修剪算法来满足其需求。

结论
----------

| 决策树使您可以快速有效地对数据进行分类。
| 因为它们将数据塑造为决策的层次结构，所以即使非专家也可以很好地理解它们。
| 决策树是通过两步过程创建和完善的：
1. 归纳（induction）
2. 修剪（pruning）。
| 归纳法涉及挑选最佳属性进行拆分，而修剪则有助于过滤掉认为无用的结果。
| 由于决策树非常易于创建和理解，因此通常是用于建模和预测数据集结果的第一种方法。

代码示例
------------

| 所提供的代码 `decisiontrees.py`_ 将采用本文档中讨论的示例，并从中创建一个决策树。
| 首先，定义每个类的每个可能选项。
| 稍后将其用于适应和显示我们的决策树：


.. _decisiontrees.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/DecisionTree/decisiontrees.py

.. code-block:: python
            import graphviz
            import itertools
            import random 
            from sklearn.tree import DecisionTreeClassifier, export_graphviz
            from sklearn.preprocessing import OneHotEncoder

            # The possible values for each class 
            classes = {
                'supplies': ['low', 'med', 'high'],
                'weather':  ['raining', 'cloudy', 'sunny'],
                'worked?':  ['yes', 'no']
            }

            # Our example data from the documentation
            data = [
                ['low',  'sunny',   'yes'],
                ['high', 'sunny',   'yes'],
                ['med',  'cloudy',  'yes'],
                ['low',  'raining', 'yes'],
                ['low',  'cloudy',  'no' ],
                ['high', 'sunny',   'no' ],
                ['high', 'raining', 'no' ],
                ['med',  'cloudy',  'yes'],
                ['low',  'raining', 'yes'],
                ['low',  'raining', 'no' ],
                ['med',  'sunny',   'no' ],
                ['high', 'sunny',   'yes']
            ]

            # Our target variable, whether someone went shopping
            target = ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']

            # Scikit learn can't handle categorical data, so form numeric representations of the above data
            # Categorical data support may be added in the future: https://github.com/scikit-learn/scikit-learn/pull/4899
            categories = [classes['supplies'], classes['weather'], classes['worked?']]
            encoder = OneHotEncoder(categories=categories)

            x_data = encoder.fit_transform(data)

            # Form and fit our decision tree to the now-encoded data
            classifier = DecisionTreeClassifier()
            tree = classifier.fit(x_data, target)

            # Now that we have our decision tree, let's predict some outcomes from random data
            # This goes through each class and builds a random set of 5 data points
            prediction_data = []
            for _ in itertools.repeat(None, 5):
                prediction_data.append([
                    random.choice(classes['supplies']),
                    random.choice(classes['weather']),
                    random.choice(classes['worked?'])
                ])

            # Use our tree to predict the outcome of the random values
            prediction_results = tree.predict(encoder.transform(prediction_data))



            # =============================================================================
            # Output code

            def format_array(arr):
                return "".join(["| {:<10}".format(item) for item in arr])

            def print_table(data, results):
                line = "day  " + format_array(list(classes.keys()) + ["went shopping?"])
                print("-" * len(line))
                print(line)
                print("-" * len(line))

                for day, row in enumerate(data):
                    print("{:<5}".format(day + 1) + format_array(row + [results[day]]))
                print("")

            feature_names = (
                ['supplies-' + x for x in classes["supplies"]] +
                ['weather-' + x for x in classes["weather"]] +
                ['worked-' + x for x in classes["worked?"]]
            )

            # Shows a visualization of the decision tree using graphviz
            # Note that sklearn is unable to generate non-binary trees, so these are based on individual options in each class
            dot_data = export_graphviz(tree, filled=True, proportion=True, feature_names=feature_names) 
            graph = graphviz.Source(dot_data)
            graph.render(filename='decision_tree', cleanup=True, view=True)

            # Display out training and prediction data and results
            print("Training Data:")
            print_table(data, target)

            print("Predicted Random Results:")
            print_table(prediction_data, prediction_results)


.. code:: python

    # The possible values for each class
    classes = {
        'supplies': ['low', 'med', 'high'],
        'weather':  ['raining', 'cloudy', 'sunny'],
        'worked?':  ['yes', 'no']
    }

| 
| 接下来，我们创建了上面显示的数据集的矩阵，并定义了每一行的结果：
| 


.. code:: python

    # Our example data from the documentation
    data = [
        ['low',  'sunny',   'yes'],
        ['high', 'sunny',   'yes'],
        ['med',  'cloudy',  'yes'],
        ['low',  'raining', 'yes'],
        ['low',  'cloudy',  'no' ],
        ['high', 'sunny',   'no' ],
        ['high', 'raining', 'no' ],
        ['med',  'cloudy',  'yes'],
        ['low',  'raining', 'yes'],
        ['low',  'raining', 'no' ],
        ['med',  'sunny',   'no' ],
        ['high', 'sunny',   'yes']
    ]

    # Our target variable, whether someone went shopping
    target = ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']


| 不幸的是，sklearn机器学习程序包无法根据分类数据创建决策树。
| 有正在进行的工作允许这样做，但是现在我们需要另一种方法来用库在决策树中表示数据。
| 天真的方法是只枚举每个类别-例如，将sunny/raining/cloudy转换为0、1和2之类的值。
| 但是这样做有一些不幸的副作用，例如这些值是可比较的（sunny<raining）并且连续。
| 为了解决这个问题，我们对数据进行“独热编码”（one hot encode）：

.. code:: python

    categories = [classes['supplies'], classes['weather'], classes['worked?']]
    encoder = OneHotEncoder(categories=categories)

    x_data = encoder.fit_transform(data)

| 
| 独热编码（One hot encoding）使我们能够将分类数据转换为ML算法可识别的，期望连续数据的值。
| 它通过选择一个类并将其划分为每个选项来工作，其中一点代表该选项是否存在。
| 
| 现在我们有了适合sklearn决策树模型的数据，我们只需将分类器拟合到数据即可：


.. code:: python

    # Form and fit our decision tree to the now-encoded data
    classifier = DecisionTreeClassifier()
    tree = classifier.fit(x_data, target)

| 
| 其余代码涉及创建一些随机预测输入，以显示如何使用树。
| 我们创建与上述数据相同格式的随机数据集，然后将其传递给DecisionTreeClassifier的预测方法。
| 这为我们提供了一系列预测目标变量-在这种情况下，对Mike是否会购物的答案是或否：

.. code:: python

    # Use our tree to predict the outcome of the random values
    prediction_results = tree.predict(encoder.transform(prediction_data))


参考资料
----------

1. https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052
2. https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e23 
3. https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ 
#. https://sebastianraschka.com/faq/docs/decision-tree-binary.html
#. https://www.cs.cmu.edu/~bhiksha/courses/10-601/decisiontrees/



