# NLP 知识点总结

## Header

参考资料：NLP的巨人肩膀（上）、（中）、（下）
_Reference:https://www.jianshu.com/p/fa95963c9abd_
_Reference:https://www.jianshu.com/p/81dddec296fa_
_Reference:https://www.jianshu.com/p/922b2c12705b_

感谢这篇文章的作者，我在学习NLP的过程中，一开始是很零散化的学习，东拼拼，西凑凑，在Standford的CS224N视频中从word2vec到rnn，然后再到lstm，接着是seq2seq，再后来是attention和Transformer等.学完之后也依然很迷糊，对概念有了一个大致的了解，但是问到具体的原理和数学推导，就懵逼了。因此我从新梳理了一下。根据这系列文章，我一步步的补全我的nlp知识，并作出了如下笔记。该笔记会随着我的学习，不断完善，也希望大家可以和我一起完善这个总结。

同时感谢我的好友林浩星，在我学习Machine Learning、NLP、RS的过程中，耐心的讲解原理，和提供了许多的参考资料让我去深入理解这些知识原理。



## Language Model 语言模型

**本质：**对一段自然语言的文本进行预测概率的大小。语言模型可以提供词表征和单词序列的概率化表示。

因为语言文字是人类认知过程中产生的高层次认知抽象实体，所以我们需要将其转换成计算机可以理解的方式，这个过程称之为文本表示。

文本表示的种类有三种：
1. 离散表示（统计语言模型）：one-hot, 词袋模型（bag-of-words), tf-idf,n-gram
2. 分布式表示:共现矩阵
3. 神经网络表示：NNLM, word2vec, glove, Elmo,BERT

给定一段话S=w_1w\_2w\_3...
P(S)=P(w\_1,w\_2,w\_3...)=P(w\_1)P(w\_2,w\_3...|w\_1)P(w\_3...|w\_1,w\_2)...

但是这样的模型有很严重的问题：

1. 很多词语同时出现的可能性太多，无法估算(**自由参数数目**)；
2. 对非常多词语的组合在语料库中并未出现，这导致了概率可能为0，使数据稀疏严重(**数据稀疏性**)。

为此引入了马尔科夫假设，每一个词出现的概率只与前一个词或前几个有限的词有关。

## Statistical Language Model 统计语言模型

> 统计语言模型采用**最大对数似然**来作为目标函数，目标是求解每一个条件概率的值。

### n-gram

N-gram语言模型的目标跟传统统计语言模型一致，都是求解每一个条件概率的值，简单计算N元语法在语料中出现的频率，然后归一化。

1. **unigram:**如果每个词都是独立的，那么我们称之为一元语言模型(考虑不到两个词语之间的关系搭配)
   P(S)=P(w\_1) * P(w\_2) * P(w\_3)...
2. **bi-gram:**如果每个词都只与它的前一个词有关，那么我们称之为二元语言模型
   P(S)=P(w\_1)P(w\_2|w\_1)P(w\_3|w\_2)...
3. **tri-gram:**如果每个词都只与它的前两个词有关，那么我们称之为三元语言模型
   P(S)=P(w\_1)P(w\_2|w\_1)P(w\_3|w\_2,w\_1)...

> 一般来说，N元模型就是假设当前词的出现概率只与它前面的N-1个词有关，且对N>1的模型，会加上句首开头标识和句末标识。

从模型的效果来看，理论上N的取值越大，效果越好。但随着N取值的增加，效果提升的幅度是在下降的。同时还涉及到一个可靠性和可区别性的问题，参数越多，可区别性越好，但同时单个参数的实例变少从而降低了可靠性。

- N-gram 解决了自由参数数目的问题。
- 可以通过平滑化解决数据稀疏性的问题，如加一平滑(拉普拉斯平滑)

**词向量:**用来表示词的向量，也可以被认为是**词的特征向量或表征**
**词嵌入:**把词语映射为实数域向量的技术。将one-hot的词向量嵌入到一个低维空间
**词干化（stemming):**就是去除词缀得到词根的过程。

### One-hot 独热编码

将词表示为只含有0-1的向量，维度与词语数|V|一致，当前表示的词语的值为1，其他为0，比如一个句子含有100个唯一且不相同的词，那么这个词向量的维度是100，若表示第3个词，则这个词向量只有第三维度是1，其他均为0.

缺点:当文本数大的时候，one-hot向量表示的维度就变得很大。

### Bag-of-words 词袋模型

词袋模型就是将所有词语装进一个袋子里，不考虑其词法和语序的问题，即每个词语都是独立的。即通过一个向量列表示一个句子，每一位表示一个单词，每一位的值表示该单词出现的频率（次数）。

缺点：并不考虑语序关系。

### TF-IDF

TF-IDF是一种用于信息检索与数据挖掘的常用加权技术，常用于挖掘文章中的关键词，而且算法简单高效，常被工业用于最开始的文本数据清洗。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章.

- tf(Term Frequency)表示词频，表示某个单词在该文本中出现的次数。
  $$ TF = (\frac{词语i在文本j中出现的次数}{文本的总次数})\ $$

  注释：有时候，tf也用$ (\frac{词语i在文本j中出现的次数}{文本j中出现最多的词语频率})\ $来表示

- idf(Inverse Document Frequency)：逆文本频率，如果包含词条 t的文档越少,IDF越大，则说明词条具有很好的类别区分能力。
  文档频率: df = 该词在所有文本中(corpus)出现的次数，比如文本库中共有4个文本，其中两个文本出现了”love“这个词，所以df("love")=2
  $$ IDF = log(\frac{N}{df+1}) \  (N是指文本库中的总文本数) $$
  $$ TF-IDF = TF*IDF $$



优点:简单快速，而且容易理解。

缺点:有时候用词频来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。

## 分布式表示

### 共现矩阵

**TO-DO:补充**



### GloVe(Global Vectors for Word Representation)

它是一个基于全局词频统计（count-based & overall statistics）的词表征工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性、类比性等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

训练方法：首先基于语料库建立共现矩阵，然后通过共现矩阵和GloVe模型学习词向量。

代价函数：$J = \sum_{i,j}^Nf(X_{i,j})(V_i^TV_j+b_i+b_j-log(X_{i,j}))^2$

$X_i\,_j$是共现矩阵中,单词i和j同时出现的值

$X_i$是共现矩阵中，单词i所在的行的和，即 $X_i = \sum_{j=1}^NX_{i,j}$
单词k出现在单词i的语境中的概率:$P_{i,k}=\frac{X_{i,k}}{X_i}$
两个单词条件概率的比率: $Ratio_{i,j,k} = \frac{P_{i,k}}{P_{i,j}}$

_Reference-《理解GloVe模型》:https://blog.csdn.net/coderTC/article/details/73864097?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522158622978519725256752313%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=158622978519725256752313&biz_id=14&utm_source=distribute.pc_search_result.none-task-blog-soetl_SOETL-2_

## Neural Network Language Model

> NNLM一般用**最大对数似然**或**KL散度**作为损失函数。

比较复杂，未考虑上下文顺序关系，将前w-1个词当做输入。计算量大，训练效果低。

### word2vec
word2vec是在NNLM上的突破，实际上并不算是一个深度学习神经网络，因为事实上它只有两层。word2vec也并不算得上一个模型，而是一个工具，真正的模型是word2vec后面的skip-gram和CBOW(Continue Bag-Of-Words)模型。

Skip-gram和CBOW的模型结构：
<img src="/Users/Simonchan/Documents/NLP/NLP/pic/word2vec.png" alt="word2vec" style="zoom:50%;" />



Word2Vec采用了negative sampling, hierachical softmax等技术对训练进行了优化，这两个操作都是在softmax输出层上进行优化。同时，也基于NEG和H-Softmax两种方式对两个神经网络进行搭建。（注：这两种技术都可以用在skip-gram和CBOW中，但是不能同时使用）

- **负例采样(negative sampling):** 对于我们在训练原始文本中遇到的每一个单词，它们都有一定概率被我们从文本中删掉，而这个被删除的概率与单词的频率有关。删除的概率为(sqrt(x/threshold)+1)*(threshold/x),其中x为词频。词语数量|V|的大小决定了训练神经网络的时候会拥有大规模的权重矩阵W，所有的这些权重矩阵需要数以亿计的样本进行调整，这不仅很消耗计算资源，而且在实际训练中效率非常慢。负采样每次只让一个样本更新仅仅更新一小部分的权重，从而降低计算量。除了当前单词，其他单词被当做是negative sample，一个单词被选作negative sample的概率跟它出现的频次有关，出现频次越高的单词越容易被选作negative words。被选做negative word的概率为x^(3/4)/∑x^(3/4),x代表词频。

- **层次Softmax(Hierachical Softmax):** 采用哈夫曼树的树形结构，同时采用logistics function作为路径二分类的概率分布输出。表示输出，这样可以大大的减少Softmax的计算量，从O(N)变成O(log(N)。

  

TO-DO:补充推导笔记图片**



> 假设一个句子S由n个词组成

从直观上理解**Skip-gram**是通过输入第i个词(center)，1<=i<=n，来预测其他词语，也就是上下文(context)出现的概率，而**CBOW**是通过上下文的词语，来估算当前词语i的语言模型。

**skip-gram过程:**先将当前单词转成one-hot词向量，然后随机生成一个Vocabulary_size*embeding_dim大小的权重矩阵，通过one-hot向量与权重矩阵相乘，得到当前词语向量中的权重向量，然后跟其他词语的one-hot向量相乘，得到的数值再通过softmax将其归一化，最后得到每个词的概率分布，最后根据这个概率分布，得到最大的那个值就是我们要输出的单词的下标。

**CBOW过程:**先将上下文单词转换成one-hot向量，然后通过一个初始化权重矩阵将向量的值相加后求平均值，之后再乘以一个输出权重矩阵计算出一个概率值去预测center值的概率分布。


_Reference-《word2vec中的数学原理详解》:https://www.cnblogs.com/peghoty/p/3857839.html_



### FastText
结构与word2vec相似，但是输入的不再局限于词向量， 也可以结合n-gram的语言信息，同时输出也不再是单个词的概率分布，而是文本分类。所以这是一个监督模型



### RNN (Recurrent Neural Network 循环神经网络)
RNN是一种特殊的神经网络，根据”人的认知是基于过往的经验和记忆“这一观点提出的。RNN之所以叫循环神经网络，是因为一个序列当前的输出跟前面的输出有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。
#### 网络结构
<img src="/Users/Simonchan/Documents/NLP/NLP/pic/RNN.jpg" alt="RNN" style="zoom:30%;" />

### RNN的前向神经网络：

$$Hidden layer: h_t = F(W*h_{t-1}+U*x_1)$$
$$output layer: y_t = G(h_t)$$
其中涉及到了三个权重矩阵，分别是：

- 从Input layer到Hidden layer的权重矩阵U
- 从t-1时刻的Hidden layer状态结点到t时刻状态结点的权重矩阵W
- 从Hidden layer到Output layer的权重矩阵V

#### BPTT(Back Propagation Through Time):
因为每一个Output layer和Hidden layer都与该结点的前一个结点有关，所以通过BP仅对当前时间t进行梯度下降是不行的。
**TO-DO：补充BPTT计算推导**
**优点:**计算长时间序列的效果不错
**缺点:**由于BP和长时间记忆的问题，会出现梯度消失或者梯度下降


_Reference:https://blog.csdn.net/qq_32241189/article/details/80461635_

作为RNN的改进，引入了两种门控技术,为LSTM和GRU。

- 对于梯度消失:由于它们都有特殊的方式存储”记忆”，那么以前梯度比较大的”记忆”不会像简单的RNN一样马上被抹除，因此可以一定程度上克服梯度消失问题。
- 对于梯度爆炸:用来克服梯度爆炸的问题就是gradient clipping，也就是当你计算的梯度超过阈值c或者小于阈值-c的时候，便把此时的梯度设置成c或-c。

### LSTM(Long Short Term Memory Network 长短时记忆网络)

_Reference:https://www.jianshu.com/p/dcec3f07d3b5_
_Reference:http://colah.github.io/posts/2015-08-Understanding-LSTMs/_

### GRU

**To-Do:补充**


### Seq2Seq 
用传统的RNN或者LSTM等语言模型，会出现一个问题，就是输入输出不等长问题，为了解决这个问题，提出了Encoder-Decoder框架思想，Seq2Seq就是Encoder-Decoder架构中最出名的模型。
结构：将两个RNN模型（或者是其他语言模型）拼接起来，一个作为Encoder，最终输出一个**语义向量**，并且将这个语义向量当做Decoder中的Hidden layer的初始状态输入。另一个Decoder接收原本的词向量，结合接收到的语义向量，输出目标。

### Attention Mechanism 注意力机制
目前大多数注意力模型附着于Encoder-Decoder框架下，目标是文本对齐

Attention值计算三步骤：
1. 计算query和key的相似度（点积、cosine等）
2. 对相似度通过Softmax进行归一化变成权重系数
3. 对权重系数和value的值进行加权求和

- Soft Attention
- Self Attention

_Reference-张俊林-《深度学习中的注意力机制》:https://blog.csdn.net/malefactor/article/details/78767781_

### ELMo(Embedding from Language Model)
思想是先预训练好，然后fine-tune
三个模型框架整合，输入和输出层都是CNN模型，不过该CNN的输入粒度是字符。中间是由Bi-LSTM组成的。

**神经网络结构:**
<img src="/Users/Simonchan/Documents/NLP/NLP/pic/ELMo_1.png" alt="ELMo_1" style="zoom:75%;" />

### GPT

### Transformer
最核心的机制就是self-attention,与传统的attention机制不同，传统的attention机制关心的是输入和输出的序列之间联系，而self-attention更关注的是序列内部词语之间的联系。

#### self-attention 自注意力



#### multi-headed attention 多头注意力机制

_Reference-《BERT大火却不懂Transformer？读这一篇就够了》:https://zhuanlan.zhihu.com/p/54356280?utm_source=wechat_session&utm_medium=social&utm_oi=851764705579110400_

#### fine-tuning 微调技巧

1. discriminative fine-tuning,
2. slanted triangular learning rates
3. gradual unfreezing

### BERT





