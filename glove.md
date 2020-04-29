# GloVe

### GloVe\(Global Vectors for Word Representation\)

它是一个基于全局词频统计（count-based & overall statistics）的词表征工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性、类比性等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

训练方法：首先基于语料库建立共现矩阵，然后通过共现矩阵和GloVe模型学习词向量。

代价函数：$$J = \sum_{i,j}^Nf(X_{i,j})(V_i^TV_j+b_i+b_j-log(X_{i,j}))^2$$

$$X_i\,_j$$是共现矩阵中,单词i和j同时出现的值

\_\_$$X_i$$是共现矩阵中，单词i所在的行的和，即:

$$X_i = \sum_{j=1}^NX_{i,j}$$ 

单词k出现在单词i的语境中的概率:$$P_{i,k}=\frac{X_{i,k}}{X_i}$$ 

两个单词条件概率的比率: $$Ratio_{i,j,k} = \frac{P{i,k}}{P{i,j}}$$

_Reference-《理解GloVe模型》:_[https://blog.csdn.net/coderTC/article/details/73864097?ops\_request\_misc=%257B%2522request%255Fid%2522%253A%2522158622978519725256752313%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request\_id=158622978519725256752313&biz\_id=14&utm\_source=distribute.pc\_search\_result.none-task-blog-soetl\_SOETL-2](https://blog.csdn.net/coderTC/article/details/73864097?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522158622978519725256752313%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=158622978519725256752313&biz_id=14&utm_source=distribute.pc_search_result.none-task-blog-soetl_SOETL-2)

