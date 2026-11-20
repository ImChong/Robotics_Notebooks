# 深蓝AI：深度学习的四大天王——DNN、CNN、RNN、GAN

> 资料归档（中文科普文章；此文件是提炼与核对记录，不转载全文或配图）

- **标题：** 深度学习的四大天王：DNN、CNN、RNN、GAN
- **作者 / 发布者：** 深蓝AI（文章署名编辑：咖啡鱼；审核：阿蓝）
- **类型：** blog / 公众号科普
- **发布时间：** 2026-11-20 17:32（文章提供的时间）
- **原文链接：** <https://mp.weixin.qq.com/s/k2axsLg86ugvFPjJohtKmQ>
- **入库日期：** 2026-11-20
- **素材方式：** 用户在当前 ingest 消息中提供文章的完整文字主体、各节标题、图片图注和原始链接；没有提供图片文件。网页读取工具访问该链接失败，因此未核验图像内容及微信页面元信息。
- **核对范围：** 已逐节阅读用户提供的全文文字：引言、1 DNN、2 CNN、3 RNN、4 GAN、5 总结及图注；核对其中机制、参数量举例、GAN 分类和现代架构关系，并使用已有一手论文索引与公开论文原文页交叉验证。图像内容、微信公众号页面实际渲染未核验。
- **通读：** 已完成
- **资料性质：** 这是一篇二次科普，不是 DNN/CNN/RNN/GAN 的首发论文；引用时区分作者观点与经典研究结论。

## 核心摘录（面向 wiki 编译）

1. **分类轴先分开。** 文章试图说明 DNN/MLP、CNN、RNN 主讲计算结构与归纳偏置，而 GAN 主讲由生成器与判别器组成的对抗学习框架；不要直接问「GAN 和 CNN 谁更强」。
2. **DNN/MLP。** 仿射层接非线性，使用链式法则反传与梯度优化。万能逼近是存在性结论，不保证优化成功、参数高效或泛化。须修正原文等同说法：DNN 是深层神经网络广义称谓，不专指全连接 MLP；CNN/RNN 也可属于 DNN。
3. **CNN。** 局部连接与空间权值共享形成归纳偏置；例如 224×224×3 → 1000 全连接是 150,528,000 个权重（不含偏置），3×3、3 输入通道、64 输出通道的卷积为 1,728 个权重加 64 个偏置，即 1,792 个参数。这是两个不同输出结构的示意量级比较，不能推断同等精度或同等计算量。标准卷积主要具平移等变性，分类不变性需要池化/聚合/训练等机制；展平不会丢失全部坐标信息，只是失去明确空间局部归纳偏置。
4. **RNN。** 同参数跨时间递推，BPTT 有梯度消失/爆炸风险；LSTM 通过带门控的细胞状态建立较好的长程梯度通路，GRU 用两门简化。序列推理递推仍常用于流式/部分可观测任务；训练时间步并行受限。
5. **GAN。** 生成器 G 和判别器 D 的对抗极小极大训练，可使用 MLP、CNN、RNN 等不同结构。原始模型理想平衡满足 p_g=p_data、最优 D=1/2；这有理论条件，不能作为任意训练过程保证。模式崩塌、梯度饱和与训练不稳是实践风险；DCGAN、cGAN、StyleGAN 是不同扩展。
6. **时代叙事不等于继承树。** Transformer、扩散模型等改变了主流应用格局，但 CNN、RNN 和对抗训练仍在具体任务中使用，不宜用「淘汰」作为无条件技术判断。
7. **机器人映射。** 向量化状态→MLP 控制；图像→CNN 编码；时间观测→GRU/RNN；对抗运动先验 AMP 借鉴判别式训练信号，但 AMP 不等同于原始图像生成 GAN。

## 术语与原文纠偏

- 原文「DNN 是所有东西的底座 / DNN 等于 MLP」属于面向初学者的简化；严格分类中深度网络是一个上位概念，全连接只是结构之一。
- 原文「GAN 只规定目标函数」需补充：它还定义一套生成器—判别器的训练交互框架；不规定 G/D 内部使用哪种骨干。
- 原文「CNN 平移等变」与「猫在哪里都还是猫（不变）」应区别：特征随平移移动 vs 分类输出保持一致。
- 原文「前三者解决判别问题」过于绝对；CNN/RNN/MLP 同样可用在生成器、编码器、回归器和策略网络中。
- 原文「GAN 用随机噪声生成文本」仅是概念性举例；离散文本生成通常需要额外的优化处理，不是原始 GAN 直接适配。

## 核对所用一手资料

- [LeCun 等：Gradient-Based Learning Applied to Document Recognition (1998)](https://doi.org/10.1109/5.726791)
- [Hochreiter & Schmidhuber：Long Short-Term Memory (1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Goodfellow 等：Generative Adversarial Nets (NeurIPS 2014)](https://papers.nips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)
- [Deep Learning（Goodfellow, Bengio, Courville）](https://www.deeplearningbook.org/)

## 对 wiki 的映射

- [四者分类维度对照](../../wiki/comparisons/dnn-cnn-rnn-gan-architecture-vs-training.md)
- [MLP](../../wiki/concepts/mlp.md)
- [CNN](../../wiki/concepts/convolutional-neural-network.md)
- [RNN](../../wiki/concepts/recurrent-neural-network.md)
- [GAN](../../wiki/concepts/generative-adversarial-network.md)
- [模型架构纵深路线 Stage 1](../../roadmap/depth-model-architecture.md)
