##########################
朴素贝叶斯分类(Naive Bayes Classification)
##########################

.. contents::
  :local:
  :depth: 3


**********
动机
**********
| 机器学习中经常出现的问题是需要将输入分类为某个预先存在的class。
| 考虑以下示例。


Say we want to classify a random piece of fruit we found lying around. In this
example, we have three existing fruit categories: apple, blueberry, and
coconut. Each of these fruits have three features we care about: size, weight,
and color. This information is shown in *Figure 1*.
| 假设我们要对发现的随机分布的水果进行分类。
| 在此示例中，我们有三个现有的水果类别：苹果，蓝莓和椰子(pple, blueberry, and coconut)
| 这些水果中的每一个都有我们关注的三个特征：大小，重量和颜色。
| 此信息如图1所示。

.. csv-table:: **Figure 1. A table of fruit characteristics**
   :header: "", "Apple", "Blueberry", "Coconut"
   :stub-columns: 1

   "Size", "Moderate", "Small", "Large"
   "Weight", "Moderate", "Light", "Heavy"
   "Color", "Red", "Blue", "Brown"


| 我们观察发现的那块水果，确定其大小适中，较重且为红色。
| 我们可以将这些功能与已知类的功能进行比较，以猜测它是哪种水果。
| 未知的水果像椰子一样重，但与苹果类具有更多的共同特征。
| 未知水果与苹果类共有3个特征中的2个，因此我们猜测它是一个苹果。
| 我们使用随机水果大小适中且像苹果一样红色的事实来猜测。
| 
| 这个例子有点愚蠢，但是它突出了有关分类问题的一些基本要点。
| 在这些类型的问题中，我们正在将未知输入的特征与数据集中已知类的特征进行比较。
| 朴素贝叶斯分类是实现此目的的一种方法。


***********
它是什么？
***********

| 朴素贝叶斯是一种分类技术，它使用我们已经知道的概率来确定如何对输入进行分类。
| 这些概率与现有类及其具有的功能有关。
| 在上面的示例中，我们选择与输入最相似的类作为其分类。
| 该技术基于使用贝叶斯定理。
| 如果您不了解贝叶斯定理是什么，请不要担心！我们将在下一部分中对其进行解释。


**************
贝叶斯定理(Bayes’ Theorem)
**************

贝叶斯定理[*Equation 1*] 是非常有用的结果，在概率论和其他学科中都得到了证明。

.. figure:: _img/Bayes.png

   **Equation 1. Bayes' Theorem**


| 利用贝叶斯定理，我们可以检查条件概率（在发生另一个事件的情况下事件发生的概率）。
| P（A | B）是在事件B已经发生的情况下事件A将发生的概率。
| 我们可以使用已知的有关事件A和事件B的其他信息来确定该值。
| 我们需要知道P（B | A）（假设事件A已经发生，事件B发生的概率），P（B）（概率事件B将发生）和P（A）（事件A将发生的概率）。
| 我们甚至可以将贝叶斯定理应用于机器学习问题！


***********
朴素贝叶斯(Naive Bayes)
***********

| 朴素贝叶斯分类使用贝叶斯定理和一些其他假设。
| 我们将假设的主要特征是功能是独立的。
| 假设独立性意味着给定特定类别出现一组特征的概率与给定该类别出现每个单个特征的所有概率的乘积相同。
| 在上面的水果示例中，红色不会影响中等大小的可能性，因此假设颜色和大小之间的独立性很好。
| 在要素可能具有复杂关系的现实世界问题中通常不是这种情况。
| 这就是为什么“naive”是名字的原因。
| 如果数学看起来很复杂，请不要担心！该代码将为我们处理数字运算。请记住，我们假设要素彼此独立以简化计算。
| 
| 在这项技术中，我们接受一些输入，并计算它发生的可能性，因为它属于我们的类别之一。我们必须对每个classes都这样做 。
| 在获得所有这些概率之后，我们仅将最大的概率作为我们对输入所属类别(classes)的预测。


**********
算法(Algorithms)
**********

| 以下是用于朴素贝叶斯分类的一些常见模型。
| 根据使用的特征分布类型，我们将它们分为两种一般情况：
1.连续(continuous)-----连续表示实数值(real-valued)（可以有十进制(decimal)答案）
2.离散(discrete)-------离散表示计数（只能有整数(whole number)答案）
| 还提供了每种算法的相关代码段。

高斯模型（连续）Gaussian Model (Continuous)
===========================

| 高斯模型假设特征（features）服从正态分布（normal distribution）。
| 据您所知，正态分布只是概率值的一种特定类型，其中值趋于接近平均值。
| 正如你可以看到 图2，正态分布的情节有一个钟形。
| 值在图的峰值附近最频繁，并且越远越难得。
| 这是另一个很大的假设，因为许多功能未遵循正态分布。
| 虽然这是事实，但假设正态分布会使我们的计算变得容易得多。
| 当特征不计数且包含十进制值时，我们使用高斯模型。

.. figure:: _img/Bell_Curve.png 

   **Figure 2. A normal distribution with the iconic bell curve shape** 
   [`code`__]
   
   .. __: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/Naive_Bayes/bell_curve.py

相关代码可在 gaussian.py_ 文件中找到。

.. _gaussian.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/Naive_Bayes/gaussian.py


| 在代码中，我们尝试从给定的RGB百分比猜测颜色。
| 我们创建一些数据来处理，其中每个数据点代表一个RGB三元组。
| 三元组的值是从0到1的十进制数，每个值都有与之关联的颜色类别。
| 我们创建一个高斯模型并将其拟合到数据中。
| 然后，我们使用新的输入进行预测，以查看应将其分类为哪种颜色。

多项式模型（离散）Multinomial Model (Discrete)
============================
Multinomial models are used when we are working with discrete counts.
Specifically, we want to use them when we are counting how often a feature
occurs. For example, we might want to count how often the word “count” appears
on this page. *Figure 3* shows the sort of data we might use with a 
multinomial model. If we know the counts will only be one of two values, we 
should use a Bernoulli model instead.
| 当我们处理离散计数（discrete counts）时，将使用多项式模型。
| 具体来说，我们在计算功能出现的频率时要使用它们。
| 例如，我们可能想计算“count(计数)”一词在此页面上出现的频率。
| 图3显示了我们可能在多项模型中使用的数据类型。
| 如果我们知道计数将只是两个值之一，则应改用Bernoulli模型。

.. csv-table:: **Figure 3. A table of word frequencies for this page**
   :header: "Word", "Frequency"
   :stub-columns: 1

   "Algebra", "0"
   "Big", "1"
   "Count", "2"
   "Data", "12"

相关的代码位于 multinomial.py_ file文件中。

.. _multinomial.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/Naive_Bayes/multinomial.py


| 该代码基于我们的水果示例。在代码中，我们尝试从给定的特性中猜出一个结果。
| 我们创建一些数据以供处理，其中每个数据点都是代表水果特征（即大小，重量和颜色）的三元组。
| 三元组的值是介于0到2之间的整数，并且每个都有与之关联的水果类。
| 整数基本上只是与特征相关的标签，但是使用它们而不是字符串可以使我们使用多项模型。
| 我们创建一个多项模型并将其拟合到数据中。
| 然后，我们使用新的输入进行预测，以查看应将其分类为哪种水果。

伯努利模型（离散） Bernoulli Model (Discrete)
==========================

| 当我们处理离散计数时，也会使用伯努利模型。
| 与多项式情况不同，这里我们在计算是否发生了特征。
| 例如，我们可能要检查“ count”一词是否在此页面上全部出现。
| 当要素只有两个可能的值（例如红色或蓝色）时，我们也可以使用伯努利模型。
| 图4显示了我们可以在Bernoulli模型中使用的数据种类。

.. csv-table:: **Figure 4. A table of word appearances on this page**
   :header: "Word", "Present?"
   :stub-columns: 1

   "Algebra", "False"
   "Big", "True"
   "Count", "True"
   "Data", "True"

相关代码可在 bernoulli.py_ 文件中找到。

.. _bernoulli.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/supervised/Naive_Bayes/bernoulli.py


| 在代码中，我们尝试根据某物的某些特征来猜测某物是否为鸭子。
| 我们创建一些数据来处理，每个数据点都是代表特征的三元组：走路像鸭子一样，说话像鸭子一样，很小。
| 三元组的值为true或false的值为1或0，并且每个值为鸭子或不是鸭子。
| 我们创建一个伯努利模型并将其拟合到数据中。
| 然后，我们用新的输入进行预测，以查看它是否是鸭子。


**********
结论
**********

| 在本模块中，我们了解了朴素贝叶斯分类。
| 朴素贝叶斯分类使我们可以根据现有类和要素的概率对输入进行分类。
| 如代码中所示，您不需要大量的培训数据就可以使Naive Bayes有用。
| 另一个好处是速度，它可以用于实时预测。
| 我们对使用朴素贝叶斯（Naive Bayes）做出了很多假设，因此应以一粒盐作为结果。
| 但是，如果您没有太多数据并且需要快速得出结果，那么朴素贝叶斯是解决分类问题的理想选择。


************
参考文献
************

1. https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
2. https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/ 
3. https://towardsdatascience.com/naive-bayes-in-machine-learning-f49cc8f831b4
#. https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-1-theory-8b9e361897d5

