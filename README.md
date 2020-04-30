# NLP总纲

## Header

我在学习NLP的过程中，一开始是很零散化的学习，东拼拼，西凑凑，在Standford的CS224N视频中从word2vec到rnn，然后再到lstm，接着是seq2seq，再后来是attention和Transformer等。学完之后也依然很迷糊，对概念有了一个大致的了解，但是问到具体的原理和数学推导，就懵逼了。因此我从新梳理了一下，一步步的补全我的nlp知识，并作出了如下笔记。该笔记会随着我的学习，不断完善，也希望大家可以和我一起完善这个总结。

同时感谢我的好友林浩星，在我学习Machine Learning、NLP、RS的过程中，耐心的讲解原理，和提供了许多的参考资料让我去深入理解这些知识原理。

参考资料：NLP的巨人肩膀（上）、（中）、（下）

_Reference:_[https://www.jianshu.com/p/fa95963c9abd](https://www.jianshu.com/p/fa95963c9abd) _Reference:_[https://www.jianshu.com/p/81dddec296fa](https://www.jianshu.com/p/81dddec296fa) _Reference:_[https://www.jianshu.com/p/922b2c12705b](https://www.jianshu.com/p/922b2c12705b)

### 

## Neural Network Language Model

> NNLM一般用**最大对数似然**或**KL散度**作为损失函数。

比较复杂，未考虑上下文顺序关系，将前w-1个词当做输入。计算量大，训练效果低。

### FastText

结构与word2vec相似，但是输入的不再局限于词向量， 也可以结合n-gram的语言信息，同时输出也不再是单个词的概率分布，而是文本分类。所以这是一个监督模型

### Skip-thought

**To-Do:补充**

### Quick-thought

**To-Do:补充**

### GPT

**To-Do:补充**



### ERNIE \(Enhanced Representation through Knowledge Integration\)

为了中文设计的模型，结合了BERT。

**To-Do**

### Albert

**To-Do:要学**

