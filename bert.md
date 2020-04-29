# BERT



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

