##########
聚类（Clustering）
##########

.. contents::
  :local:
  :depth: 3


********
总览
********

| 在先前的模块中，我们讨论了监督学习主题。
| 现在，我们准备继续进行 **无监督学习（unsupervised learning）** ，我们的目标将有很大的不同。
| 在监督学习中，我们尝试将输入与某些现有模式进行匹配。
| 对于无监督学习，我们将尝试在未标记的原始数据集中发现模式。
| 我们看到有监督学习中经常出现分类问题，现在我们将研究无监督学习中的类似问题： **聚类（clustering）** 。


**********
聚类(Clustering)
**********

| 聚类是对相似数据进行分组并隔离相异数据的过程。
| 我们希望我们提出的集群中的数据点共享一些公共属性，以将它们与其他集群中的数据点分开。
| 最终，我们将得到一些满足这些要求的小组。
| 这听起来似乎很熟悉，因为表面上听起来很像分类。
| 但是请注意，聚类(clustering)和分类(classification)解决了两个非常不同的问题。
| 聚类用于识别数据集中的潜在组，而分类用于将输入与现有组进行匹配。


**********
动机
**********

| 聚类是解决无监督学习中常见问题的一种非常有用的技术。
| 通过聚类，我们可以通过对相似的数据点进行分组来找到数据集中的基础模式。
| 考虑玩具制造商的情况。玩具制造商生产许多产品，并且碰巧拥有多样化的消费群。
| 对于制造商来说，识别购买特定产品的群体可能很有用，以便可以个性化广告。
| 有针对性的广告是行销中的普遍愿望，而集群有助于识别人口统计信息。
| 当我们想识别原始数据集中潜在的群体结构时，聚类是一个很好的使用工具。


*******
方法
*******

| 由于聚类(clustering)只是提供了对数据集的解释，因此有许多方法可以实现它。
| 在决定群集(clusters)时，我们可能会考虑 **数据点之间的距离** 。
| 我们还可以考虑一个 **区域中的数据点密度** 以确定群集(clusters)。
| 对于此模块，我们将分析两种较常见和流行的方法：
| 

1. **K-Means**
2. **Hierarchical** 

| 
| 在这两种情况下，我们都将使用 *Figure 1* 中的数据集进行分析。

.. figure:: _img/Data_Set.png

   **Figure 1. A data set to use for clustering**
| 用于集群的数据集
| 
| 此数据集表示玩具制造商的产品数据。
| 该制造商向5至10岁的幼儿出售3种产品，
| 每种产品有3种变体。这些产品是可动人偶，积木和汽车。(action figures, building blocks, and cars)
| 制造商还指出了哪个年龄段的人群购买每种产品的比例最高。
| 数据集中的每个点代表购买最多玩具的玩具和年龄组之一。


K均值(K-Means)
=======

| K-均值聚类(K-Means clustering)尝试使用迭代过程将数据集划分为K个群集(K clusters)。
| 第一步是为每个群集(clusters)选择一个中心点。该中心点不需要与实际数据点相对应。
| 中心点可以随机选择，或者如果我们对它们应在的位置有很好的猜测，则可以选择它们。
| 在下面的代码中，中心点是使用 k-means++ 方法选择的，该方法旨在加快收敛速度(convergence)。
| 此方法的分析超出了本模块的范围，但是有关sklearn的其他初始选项，请在此处( here_)检查。

.. _here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

| 第二步是将每个数据点分配给一个群集(clusters)。
| 我们通过测量数据点和每个中心点之间的距离并选择中心点最接近的群集(clusters)来做到这一点。
| 此步骤 如 *Figure 2* 所示。

.. figure:: _img/K_Means_Step2.png

   **Figure 2. Associate each point with a cluster**
| 将每个点与一个集群相关联
| 
| 现在所有数据点都属于一个群集(clusters)，第三步是重新计算每个群集(clusters)的中心点。
| 这只是属于该群集(clusters)的所有数据点的平均值。
| 此步骤 如 *Figure 3* 所示。


.. figure:: _img/K_Means_Step3.png

   **Figure 3. Find the new center for each cluster**
| 查找每个集群的新中心
| 
| 现在，我们只重复第二和第三步，直到中心在迭代之间停止变化或仅略微变化为止。
| 结果是K个群集(K clusters)，其中数据点比任何其他群集的中心更靠近其群集的中心。
| 这在 *Figure 4* 中示出。


.. figure:: _img/K_Means_Final.png

   **Figure 4. The final clusters**
| 最终的集群
| 
| K-Means聚类要求我们输入不总是容易确定的期望聚类数。
| 取决于我们在第一步中选择起始中心点的位置，也可能会不一致。
| 在整个过程中，我们最终可能会看到集群已被优化，但可能不是最佳的整体解决方案。
| 在 *Figure 4* 中，我们以一个红色数据点结束，该数据点与红色中心和蓝色中心的距离相等。这源于我们最初的中心选择。
| 相反， *Figure 5* 显示了在给定不同起始中心的情况下可能达到的另一个结果，并且看起来更好一些。

.. figure:: _img/K_Means_Final_Alt.png

   **Figure 5. An alternative set of clusters**
| 一组替代的集群
| 
| 另一方面，K-Means功能非常强大，因为它在每个步骤都考虑了整个数据集。
| 它也很快，因为我们只计算距离。
| 因此，如果我们需要一种考虑整个数据集的快速技术，并且对底层组的外观有所了解，那么K-Means是一个不错的选择。
| 
| 相关代码位于 clustering_kmeans.py_ 文件中。
| 
.. _clustering_kmeans.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/unsupervised/Clustering/clustering_kmeans.py

| 
| 在代码中，我们创建了用于分析的简单数据集。
| 设置集群非常简单，需要一行代码：

.. code-block:: python

   kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

| 
| 选择 `n_clusters` 参数为3，因为在out数据集中似乎有3个群集。
| 每次运行代码时， `random_state` 参数都位于该位置以提供一致的结果。
| 其余代码将显示 *Figure 6* 中所示的最终图。
| 
.. figure:: _img/KMeans.png

   **Figure 6. A final clustered data set**
|    最终的集群数据集

| 
| 群集以颜色编码，'x'表示群集中心，虚线表示群集边界。

.. code-block:: python

            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from sklearn.cluster import KMeans
            from scipy.spatial import Voronoi, voronoi_plot_2d

            # This data set represents a toy manufacturer's product data
            #
            # The first value in the pair represents a toy:
            #    0-2: Action Figures
            #    3-5: Building Blocks
            #    6-8: Cars
            #
            # The second value is the age group that buys the most of that toy:
            #    0: 5 year-olds
            #    1: 6 year-olds
            #    2: 7 year-olds
            #    3: 8 year-olds
            #    4: 9 year-olds
            #    5: 10 year-olds
            x = np.array([[0,4], [1,3], [2,5], [3,2], [4,0], [5,1], [6,4], [7,5], [8,3]])

            # Set up K-Means clustering with a fixed start and stop at 3 clusters
            kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

            # Plot the data
            sns.set_style("darkgrid")
            plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, cmap=plt.get_cmap("winter"))

            # Save the axes limits of the current figure
            x_axis = plt.gca().get_xlim()
            y_axis = plt.gca().get_ylim()

            # Draw cluster boundaries and centers
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], marker='x')
            vor = Voronoi(centers)
            voronoi_plot_2d(vor, ax=plt.gca(), show_points=False, show_vertices=False)

            # Resize figure as needed
            plt.gca().set_xlim(x_axis)
            plt.gca().set_ylim(y_axis)

            # Remove ticks from the plot
            plt.xticks([])
            plt.yticks([])

            plt.tight_layout()
            plt.show()

分层(Hierarchical)
============

| 层次聚类(Hierarchical clustering)将数据集想象为群集(clusters)的层次。
| 我们可以从所有数据点中建立一个巨型群集开始。这在 *Figure 7* 中示出。

.. figure:: _img/Hierarchical_Step1.png

   **Figure 7. One giant cluster in the data set***
| 数据集中的一个巨型集群
| 
| 在此群集内，我们找到两个最不相似的子群集并将其拆分。
| 这可以通过使用一种算法来实现，以使集群间距离最大化。
| 这只是一个群集中的节点与另一群集中的节点之间的最小距离。
| 这在 *Figure 8* 中示出。



.. figure:: _img/Hierarchical_Step2.png

   **Figure 8. The giant cluster is split into 2 clusters**
| 巨型群集分为两个群集
| 
| 我们将继续拆分子群集，直到每个数据点都属于自己的群集，或者直到我们决定停止为止。
| 如果我们从一个巨型群集开始，然后将其分解为较小的群集，则称为 **自顶向下（top-down）** 或 **分裂（divisive)**  聚类（clustering）。
| 
| 或者，我们可以从考虑每个数据点的群集开始。
| 下一步是将两个最接近的群集合并为一个较大的群集。这可以通过找到每个群集之间的距离并选择它们之间距离最小的一对来完成。
| 我们将继续此过程，直到只有一个集群。
| 这种组合群集的方法称为 **自下而上（bottom-up）** 或 **凝聚（agglomerative）** 聚类（clustering）。
| 在这两种方法的任何时候，我们都可以在集群看起来合适时停止。
| 
| 与K-Means不同，分层聚类相对较慢，因此无法很好地扩展到大型数据集。
| 从好的方面来说，当您多次运行分层集群（Hierarchical clustering）时，它会更加一致，并且不需要您知道预期集群的数量。
| 
| 相关代码位于 clustering_hierarchical.py_ 文件中。


.. _clustering_hierarchical.py: https://github.com/machinelearningmindset/machine-learning-course/blob/master/code/unsupervised/Clustering/clustering_hierarchical.py


| 在代码中，我们创建了用于分析的简单数据集。
| 设置聚类（clustering）非常简单，需要一行代码：

.. code-block:: python

   hierarchical = AgglomerativeClustering(n_clusters=3).fit(x)

| 选择 `n_clusters` 参数为3，因为在out数据集中似乎有3个群集（clusters）。
| 如果我们还不知道这一点，我们可以尝试不同的值，看看哪个值最有效。
| 其余代码将显示 *Figure 9* 中所示的最终图。

.. figure:: _img/Hierarchical.png

   **Figure 9. A final clustered data set**
| 最终的群集的数据集
| 
| 群集使用颜色编码，大型群集周围有边框，以显示哪些数据点属于它们。

.. code-block:: python

            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
            from collections import defaultdict
            from scipy.spatial import ConvexHull

            # This data set represents a toy manufacturer's product data
            #
            # The first value in the pair represents a toy:
            #    0-2: Action Figures
            #    3-5: Building Blocks
            #    6-8: Cars
            #
            # The second value is the age group that buys the most of that toy:
            #    0: 5 year-olds
            #    1: 6 year-olds
            #    2: 7 year-olds
            #    3: 8 year-olds
            #    4: 9 year-olds
            #    5: 10 year-olds
            x = np.array([[0,4], [1,3], [2,5], [3,2], [4,0], [5,1], [6,4], [7,5], [8,3]])

            # Set up hierarchical clustering and stop at 3 clusters
            num_clusters = 3
            hierarchical = AgglomerativeClustering(n_clusters=num_clusters).fit(x)

            # Plot the data
            sns.set_style("darkgrid")
            colors = plt.get_cmap("winter")
            points = plt.scatter(x[:, 0], x[:, 1], c=hierarchical.labels_,
                        cmap=colors)

            # Draw in the cluster regions
            regions = defaultdict(list)
            # Split points based on cluster
            for index, label in enumerate(hierarchical.labels_):
                regions[label].append(list(x[index]))

            # If a cluster has more than 2 points, find the convex hull for the region
            # Otherwise just draw a connecting line
            for key in regions:
                cluster = np.array(regions[key])
                if len(cluster) > 2:
                    hull = ConvexHull(cluster)
                    vertices = hull.vertices
                    vertices = np.append(vertices, hull.vertices[0])
                    plt.plot(cluster[vertices, 0], cluster[vertices, 1],
                             color=points.to_rgba(key))
                else:
                    np.append(cluster, cluster[0])
                    x_region, y_region = zip(*cluster)
                    plt.plot(x_region, y_region, color=points.to_rgba(key))

            # Remove ticks from the plot
            plt.xticks([])
            plt.yticks([])

            plt.tight_layout()
            plt.show()

*******
摘要
*******

| 在本模块中，我们了解了聚类（clustering）。
| 聚类（clustering）允许我们通过对相似数据点进行分组来发现原始数据集中的模式。
| 这是无监督学习中的普遍愿望，而聚类是一种流行的技术。
| 您可能已经注意到，与以前的模块中一些数学上比较繁重的描述相比，上面讨论的方法相对简单。
| 这些方法简单但功能强大。
| 例如，我们能够确定玩具制造商示例中可用于定向广告的集群。
| 对于企业而言，这是非常有用的结果，并且只花了几行代码。
| 通过对集群的深入了解，您将为在机器学习领域取得成功做好准备。


************
参考资料
************

1. https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/
2. https://medium.com/datadriveninvestor/an-introduction-to-clustering-61f6930e3e0b
3. https://medium.com/predict/three-popular-clustering-methods-and-when-to-use-each-4227c80ba2b6
#. https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68 
#. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

