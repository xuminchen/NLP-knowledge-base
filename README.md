# NLP总纲

## Header

我在学习NLP的过程中，一开始是很零散化的学习，东拼拼，西凑凑，在Standford的CS224N视频中从word2vec到rnn，然后再到lstm，接着是seq2seq，再后来是attention和Transformer等。学完之后也依然很迷糊，对概念有了一个大致的了解，但是问到具体的原理和数学推导，就懵逼了。因此我从新梳理了一下，一步步的补全我的nlp知识，并作出了如下笔记。该笔记会随着我的学习，不断完善，也希望大家可以和我一起完善这个总结。

同时感谢我的好友林浩星，在我学习Machine Learning、NLP、RS的过程中，耐心的讲解原理，和提供了许多的参考资料让我去深入理解这些知识原理。

参考资料：NLP的巨人肩膀（上）、（中）、（下）

_Reference:_[https://www.jianshu.com/p/fa95963c9abd](https://www.jianshu.com/p/fa95963c9abd) _Reference:_[https://www.jianshu.com/p/81dddec296fa](https://www.jianshu.com/p/81dddec296fa) _Reference:_[https://www.jianshu.com/p/922b2c12705b](https://www.jianshu.com/p/922b2c12705b)





## 分布式表示

### 共现矩阵

计算两个token之间共同出现的频率

**TO-DO:补充**

### LDA 隐狄利克雷模型

利用文档中单词的贡献关系来对单词按主题聚类。

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

## Neural Network Language Model

> NNLM一般用**最大对数似然**或**KL散度**作为损失函数。

比较复杂，未考虑上下文顺序关系，将前w-1个词当做输入。计算量大，训练效果低。

### word2vec

word2vec是在NNLM上的突破，实际上并不算是一个深度学习神经网络，因为事实上它只有两层。word2vec也并不算得上一个模型，而是一个工具，真正的模型是word2vec后面的skip-gram和CBOW\(Continue Bag-Of-Words\)模型。

Skip-gram和CBOW的模型结构： 

![](.gitbook/assets/word2vec.png)



Word2Vec采用了negative sampling, hierachical softmax等技术对训练进行了优化，这两个操作都是在softmax输出层上进行优化。同时，也基于NEG和H-Softmax两种方式对两个神经网络进行搭建。（注：这两种技术都可以用在skip-gram和CBOW中，但是不能同时使用）

* **负例采样\(negative sampling\):** 对于我们在训练原始文本中遇到的每一个单词，它们都有一定概率被我们从文本中删掉，而这个被删除的概率与单词的频率有关。删除的概率为\(sqrt\(x/threshold\)+1\)\*\(threshold/x\),其中x为词频。词语数量\|V\|的大小决定了训练神经网络的时候会拥有大规模的权重矩阵W，所有的这些权重矩阵需要数以亿计的样本进行调整，这不仅很消耗计算资源，而且在实际训练中效率非常慢。负采样每次只让一个样本更新仅仅更新一小部分的权重，从而降低计算量。除了当前单词，其他单词被当做是negative sample，一个单词被选作negative sample的概率跟它出现的频次有关，出现频次越高的单词越容易被选作negative words。被选做negative word的概率为x^\(3/4\)/∑x^\(3/4\),x代表词频。
* **层次Softmax\(Hierachical Softmax\):** 采用哈夫曼树的树形结构，同时采用logistics function作为路径二分类的概率分布输出。表示输出，这样可以大大的减少Softmax的计算量，从O\(N\)变成O\(log\(N\)。

TO-DO:补充推导笔记图片\*\*

> 假设一个句子S由n个词组成

从直观上理解**Skip-gram**是通过输入第i个词\(center\)，1&lt;=i&lt;=n，来预测其他词语，也就是上下文\(context\)出现的概率，而**CBOW**是通过上下文的词语，来估算当前词语i的语言模型。

**skip-gram过程:**先将当前单词转成one-hot词向量，然后随机生成一个Vocabulary\_size\*embeding\_dim大小的权重矩阵，通过one-hot向量与权重矩阵相乘，得到当前词语向量中的权重向量，然后跟其他词语的one-hot向量相乘，得到的数值再通过softmax将其归一化，最后得到每个词的概率分布，最后根据这个概率分布，得到最大的那个值就是我们要输出的单词的下标。

**CBOW过程:**先将上下文单词转换成one-hot向量，然后通过一个初始化权重矩阵将向量的值相加后求平均值，之后再乘以一个输出权重矩阵计算出一个概率值去预测center值的概率分布。

_Reference-《word2vec中的数学原理详解》:_[https://www.cnblogs.com/peghoty/p/3857839.html](https://www.cnblogs.com/peghoty/p/3857839.html)

### FastText

结构与word2vec相似，但是输入的不再局限于词向量， 也可以结合n-gram的语言信息，同时输出也不再是单个词的概率分布，而是文本分类。所以这是一个监督模型

### Skip-thought

**To-Do:补充**

### Quick-thought

**To-Do:补充**

### RNN \(Recurrent Neural Network 循环神经网络\)

RNN是一种特殊的神经网络，2010年基于”人的认知是基于过往的经验和记忆“这一观点提出的。RNN之所以叫循环神经网络，是因为一个序列当前的输出跟前面的输出有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。

RNN有三个权重矩阵，分别是:

* 从Input layer到Hidden layer的权重矩阵U
* 从t-1时刻的Hidden layer状态结点到t时刻状态结点的权重矩阵W
* 从Hidden layer到Output layer的权重矩阵V

**这三个权重矩阵在任何时刻都是参数共享的。**

#### RNN的网络结构

![](.gitbook/assets/rnn.jpg)

#### FFNN\(Feed-Forward Neural Network\)：

$$Hidden layer: h_t = F(W*h_{t-1}+U*x_1)$$

 $$Output layer: y_t = G(h_t)$$

一般来说，F\(\)函数一般是激活函数，可以为sigmoid，也可以是ReLu等。G\(\)函数为加权求和或Softmax函数等。

> 此外，RNN的输出可以为1个，也可以为多个。当输入为M时，输出可以为1，可以为M，也可以为N\(N不等于M\)。

#### BPTT\(Back Propagation Through Time\):

因为每一个Output layer和Hidden layer都与该结点的前一个结点有关，所以通过BP仅对当前时间t进行梯度下降是不行的。

推导tips:导数乘法。

**TO-DO：补充BPTT计算推导**

优点:计算长时间序列的效果不错

缺点:由于BP和长时间记忆的问题，会出现梯度消失或者梯度爆炸。

_Reference-《深度学习之RNN\(循环神经网络\)》:_[https://blog.csdn.net/qq\_32241189/article/details/80461635](https://blog.csdn.net/qq_32241189/article/details/80461635)

作为RNN的改进，引入了两种门控技术,为LSTM和GRU。

* 对于梯度消失:由于它们都有特殊的方式存储”记忆”，那么以前梯度比较大的”记忆”不会像简单的RNN一样马上被抹除，因此可以一定程度上克服梯度消失问题。
* 对于梯度爆炸:用来克服梯度爆炸的问题就是gradient clipping，也就是当你计算的梯度超过阈值c或者小于阈值-c的时候，便把此时的梯度设置成c或-c。

### LSTM\(Long Short Term Memory Network 长短时记忆网络\)

LSTM相比传统的RNN，多了一个隐藏状态，用来保存长期的状态信息，称为**单元状态Cell state**。

那么怎么去控制这个Cell去保存的信息呢？LSTM用“门”这个概念去操控的。

LSTM有三个门，总共包含了六个计算步骤，每一个门简单的来讲就是一个全连接层，通过激活函数sigmoid之后，输出一个0-1之间的实数。

#### LSTM的网络结构：

![LSTM](.gitbook/assets/lstm.png)

> $$[h_{t-1},x_t]$$这个表示将两个向量拼接。

#### 遗忘门 forget gate:

遗忘门决定了上一时刻的单元状态有多少保留到当前时刻

$$F_t = σ(W_f*[h_{t-1},x_t]+b_f)$$

#### 输入门 input

输入门决定了当前时刻有多少信息保存到单元状态

$$i_t = σ(W_i*[h_{t-1},x_t]+b_i)$$

$$C_t^~ = tanh(W_C*[h_{t-1},x_t]+b_C)$$                 注意，这里采用tanh是为了生成一个新的Cell state

$$C_t = f_t*C_{t-1}+i_t*C_t~$$

#### 输出门 output

输出门控制单元状态有多少信息输出到隐藏状态

$$o_t = σ(W_o*[h_{t-1},x_t]+b_o)$$

$$h_t = tanh(C_t)*o_t$$                                          注意，这里采用tanh也是为了生成一个新的Hidden state

LSTM的本质还是一个RNN，所以我们依然可以通过BPTT去进行优化。

#### Bi-LSTM

另外，LSTM本身是一个left-to-right的语言模型，考虑到长句前后语义情况，提出了**Bi-LSTM**，由两个LSTM神经网络组成，一个负责left-to-right，另一个负责right-to-left，最后输出的隐状态拼接在一起后计算概率分布，然后输出。

_Reference-《详解LSTM》:_[https://www.jianshu.com/p/dcec3f07d3b5](https://www.jianshu.com/p/dcec3f07d3b5) 

_Reference-Colah- 《Understanding LSTM Networks》:_ [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### GRU\(Gate Recurrent Unit \)

GRU 和 LSTM 一样，都是RNN的优化改良版本，但是与LSTM不同的是，GRU中只用了两个门，分别是更新门\(update gate\)和重置门\(reset gate\)，而且计算开销比LSTM小很多（贫穷限制了我！）

**更新门​:**用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。

**重置门​:**控制前一状态有多少信息被写入到当前的候选集 ​上，重置门越小，前一状态的信息被写入的越少

![](.gitbook/assets/gru.png)

当前时刻的​用tanh是将当前时刻的信息缩放到\[-1,1\]

### Seq2Seq

用传统的RNN或者LSTM等语言模型处理机器翻译等工程时，会出现一个问题，就是当输入输出不等长时的映射关系，为了解决这个问题，提出了Encoder-Decoder框架思想，Seq2Seq就是Encoder-Decoder架构中最出名的模型。

**结构:**将两个RNN模型（或者是其他语言模型）拼接起来，一个作为Encoder，最终输出一个**语义向量C**，并且将这个语义向量C当做Decoder中的Hidden layer的初始状态输入。另一个Decoder接收原本的词向量，结合接收到的语义向量，输出目标。

> 关注的问题通常是输入数量为m，输出数量为n，m不等于n的情况。

#### Seq2Seq的网络结构

![](.gitbook/assets/encoder-decoder.jpg)

#### EXAMPLE

> INPUT:
>
> OUTPUT:
>
> 语义编码C=F\(x\_1,x\_2,x\_3,...,x\_m\)
>
> y\_i = G\(C,y\_1,y\_2,...,y\_i-1\)
>
> _注：F\(\)一般为激活函数，G\(\)一般为加权求和。_

从上面的例子可知，不管生成哪个单词，语义向量C都是一样的，对任何时间步来说并无区别。

所以在文本输入句子长度较短时，问题不大；当句子长度较长时，则会丢失中间单词的语义信息。

### Attention Mechanism 注意力机制

目前大多数注意力模型附着于Encoder-Decoder框架下，核心目标是从众多信息中选择出对当前任务目标更关键的信息。

不同于Seq2Seq结构，在注意力模型中，在任何时刻会分配给不同模型的注意力值大小，即概率分布。

原本的语义向量C变成根据当前生成的单词不断变化的$C\_i$

#### EXAMPLE

> $INPUT:$
>
> $OUTPUT:$
>
> $y\_i = G\(C\_i,y\_1,y\_2,...,y\_i-1\)$
>
> _注：F\(\)一般为激活函数，G\(\)一般为加权求和。_

\_\_$$C_i$$的计算公式为: $$Ci = \sum_{j=1}^{Lx}a_{i,j} h_j$$

> $L\_x$ 是句子Source的长度，a\_{i,j}$$表示输出第 i 个单词时，第 j 个单词的注意力分配系数，$h\_j$是在Encoder中 j 时刻隐层向量.

Attention值计算步骤：

1. 分别通过三个权重矩阵，将词向量生成key和value，query
2. 计算query和key的相似度（点积、cosine等）
3. 对相似度通过Softmax进行归一化变成权重系数，也就是$C_i$计算公式里的$a_{i,j}$
4. 对权重系数和value的值进行加权求和

_Reference-张俊林-《深度学习中的注意力机制》:_[https://zhuanlan.zhihu.com/p/37601161](https://zhuanlan.zhihu.com/p/37601161)

### ELMo\(Embedding from Language Model\)

RNN-based language models\(trained from lots of sentences\)

模型训练完得到contextualized word embedding.

思想是先预训练好，然后fine-tune 三个模型框架整合，输入和输出层都是CNN模型，不过该CNN的输入粒度是字符。中间是由Bi-LSTM组成的。

ELMo有很多层，每层都会生成word embedding

**神经网络结构:**

![](.gitbook/assets/elmo_1.png)

ELMo的每一层都会包含left-to-right的embedding输出和right-to-left的embedding输出，作为整一层的输出，需要对两个输出进行加权求和，这个权重W是根据我们的下游任务训练更新的。

### GPT

**To-Do:补充**

### Transformer

最核心的机制就是self-attention,与传统的attention机制不同，传统的attention机制关心的是输入和输出的序列之间联系，而self-attention更关注的是序列内部词语之间的联系。

#### self-attention 自注意力

自注意力层就是特殊的注意力层， **关注序列内部词语的联系和顺序**。计算过程与注意力层一致，不过query为Source的token。

#### multi-headed attention 多头注意力机制

* 扩展了模型专注不同位置的能力
* 提供注意力层的多个“表示子空间”（也就是会生成多个不同的Q,K,V）

1. 首先将词向量分别与 $$W_K,W_V,W_Q$$ 权重矩阵相乘n次（论文里是8次），然后会生成n个注意力头
2. 接下来将这n个注意力头左右拼接在一起（首尾相连）
3. 接着再乘以一个附加的权重矩阵 $$W_z$$ ，得到一个融合所有注意力头的矩阵z，并送入下一层的FFNN中。

![](.gitbook/assets/multi-headed_self-attention.jpg)

#### 输入-字符位置编码

为了让模型理解单词的顺序，我们添加了位置编码向量，这些向量的值遵循特定的模式。

在论文中，位置编码的生成是**由词向量的左半边通过一个正弦函数，右半边通过一个余弦函数，之后将两个输出向量拼接到一起，生成了位置编码。**

所以我们输入模型不再是一个简单的词向量，而是以**词向量+字符位置编码**作为输入。

#### add-Norm 层

在Transformer中，Encoder和Decoder都有Add-Norm层，分别在self-attention层与FFNN层之后，这个层的左右就是用来作残差相加。即把输入的词向量 x 和通过self-attention层的输出向量$z\_0$相加，然后做一个归一化处理生成向量  $$z_1$$ 。之后将 $$z_1$$ 和 $$z_1$$ 通过FFNN后的输出向量相加，再做一个归一化处理。

_Reference-《BERT大火却不懂Transformer？读这一篇就够了》:_[https://zhuanlan.zhihu.com/p/54356280?utm\_source=wechat\_session&utm\_medium=social&utm\_oi=851764705579110400](https://zhuanlan.zhihu.com/p/54356280?utm_source=wechat_session&utm_medium=social&utm_oi=851764705579110400)

#### fine-tuning 微调技巧

1. discriminative fine-tuning,
2. slanted triangular learning rates
3. gradual unfreezing

### BERT\(Bidirectional Encoder Representation from Transformers\)

多层Transformer的Encoder部分叠加而成，只采用Encoder的原因是，decoder不能获取要预测的信息

最主要的两个特点:

* Masked Language Model:捕获token级别的representation
* Next Sentence Prediction：捕获sentence级别的represention

整个模型的通俗意义就是，整个模型的下层是提取token的关系的，上层是提取sentence的关系的，中间经过多个self-attention层后，每个token之间的关系已经映射很深了，所以就相当于理解了语义关系。

#### Embedding

首先要说一下，虽然采用了Transfomer的Encoder模块，但是输入却是不尽相同的，在BERT中，除了位置编码和原本词向量外，还有一个向量就是句子编码。

分句编码由我们定义，比如`Source:[CLS]+sentence(A)+[SEP]+sentence(B)+[SEP]`,那么属于sentence\(A\)的词的分句编码就为0，属于sentence\(B\)的词的分局编码就为1。

![](.gitbook/assets/bert-embedding.jpg)

#### Masked Language Model\(MLM\)

输入的文本有一定的概率会被一个“**\[MASK\]**”遮挡，在论文中的概率是15%。

在遮挡的时候会发生以下三种情况：

* 80%会被“**\[MASK\]**”遮挡
* 10%会被替换成其他词语
* 10%会保持不变

这样就可以实现了更深层意义上的Bidirectional，因为如果仅仅是多层self-attention叠加，虽然会使当前token和全局的关系加深，但是上下文的意思依然会影响结果，因此通过一个mask，相当于我们高中时候做过的完型填空一样，我们通过训练去判断这个词的准确性。

> 在MASK中可以填的词的embedding是具有相似性的

通过BERT后的输出embedding Vector 会被丢入一个Linear Multi-class Classifier之中，要求输出的词是哪一个，这个Classifier的size是vocabularies size。

#### Next Sentence Prediction

这个同样是Bert训练过程中的模块之一，这里通过判断下一句的是否是当前句子的下一句。50%是正确的，50%是被替换掉的句子。

新增两个special token ：**\[CLS\]**和 **\[SEP\]**

* **\[CLS\]:**the position that outputs classification results.
* **\[SEP\]:**the boundary of two sentences.

**缺点：**

1. BERT非常大，导致fine-tune和训练都比较慢
2. 针对特定语言处理的问题
3. 并不是所有NLP应用方向都适用，如Language Model，text generation， translation

### ERNIE \(Enhanced Representation through Knowledge Integration\)

为了中文设计的模型，结合了BERT。

**To-Do**

### Albert

**To-Do:要学**

