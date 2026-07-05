---
title: ViT入门
author: human five
date: "2026-07-05 11:53:13"
source: "https://mp.weixin.qq.com/s/ugiOirWHrSgEefG8W1-o6Q"
---

# ViT入门

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZicf9icSrXvrzjCwiaIITnvpHarsZHoAwS6hS8WR9sqqUAdWsrO6pqCYZiaR0CqWAhibLWVicZ3JeaUqCUrom3t9xXnRGHUr0wkjicD8E/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)
> https://github.com/VizuaraAI/Transformers-for-vision-BOOK/tree/main

## 1.1 Vision Transformer 简介及与CNN对比

Vision Transformer将语言建模领域的Transformer架构迁移至计算机视觉任务。不同于卷积网络依靠小型滑动滤波器遍历图像，ViT把图像分割为多个小块，将每一块视作token，通过自注意力机制学习不同图像块之间的关联关系。简单来说，Vision Transformer能够同时观测图像全部区域，并判断不同区域间的相互影响。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZico1n5NovO95OrncbNZnIVQqjmPo9KqmlRrN2OSOhO6iaQeVODK9iaQJkJzeTQBia4ib2nIJOrI654K4dYzHf7HyicqEG5WzM1g15dA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

图1 鸟类图像上自注意力机制与卷积感受野对比

Vision Transformer与卷积网络最核心的架构差异，在于二者观测图像的方式。卷积层计算输出时，仅能读取像素的局部邻域；只有不断堆叠多层卷积与池化操作，感受野才会逐步扩大。这种局部偏置特性在传统视觉任务中效果优异，但也导致图像内远距离依赖关系只能在网络深层、间接地被捕获。图1直观展示了全局自注意力与局部卷积在鸟类图像上的区别。

ViT的自注意力在第一层就拥有全局感受野：对于任意查询位置，模型可以直接将其与图像内所有其他图像块做对比，并判断哪些区域存在关联。以图中的鸟类示意图举例，单个像素/图像块能够直接关联画面任意其他区域；而卷积只能观测周边局部区域，必须堆叠大量层才能将图像一侧的信息传递到另一侧。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/Kltic3d4ibvZ8iaQTuxSump5lsMYXNuicIR2nZU1zdhSjokrxB2psOuu4G2DwRGiciavwtjqMsibOFaovPg3FXWHm3UlESVyc3VG2WVIScAHGJttcY/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=2)

图2 视错觉绘画：全局注意力 vs 局部卷积

第二张图选用一幅视错觉绘画，更直观地体现二者差异。人类既能看清河畔骑手的细节场景，也能识别整幅画构成的人脸轮廓。卷积模型更倾向于聚焦岩石、水流、毛发等局部纹理；而Vision Transformer可以关联画面相隔很远的区域，从而识别出整体构成的人脸。图2展示了自注意力能够跨整幅画作建立关联，卷积则始终局限于局部视野。

盲人摸象的典故也能形象类比二者区别：每个盲人只能触摸大象一小块，便根据尾巴、身躯、象鼻分别得出大象是绳子、墙壁、蛇的结论。卷积网络的工作逻辑与此相似，每一个神经元仅能读取局部小块，依靠大量独立局部观测拼凑图像理解；而Vision Transformer类似一群可以自由互通信息的观测者，即便每个人初始视野有限，自注意力机制能整合全部观测结果，还原大象完整形态。

本章后续内容将从宏观对比切入，拆解Vision Transformer内部工作原理：讲解图像如何转换为图像块token、位置信息如何注入、自注意力层如何处理token序列。我们将逐层搭建完整的Vision Transformer编码器与分类头，清晰梳理整套架构。

### 从文本Transformer到Vision Transformer

文本Transformer与图像Transformer共享核心设计思想。以GPT这类语言模型为例：输入为token序列，经过嵌入层后采用掩码自注意力，每个token仅能关注自身及前文token，匹配下一词预测任务；序列最终上下文向量用于预测后续token。而Vision Transformer的流程为：先将图像切分转为图像块token序列，采用无掩码自注意力（所有token可互相关注），并引入专用class token；最终依靠该token输出特征，送入小型多层感知机头完成分类。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ8vXv9ePnMPiaTGKnYicibW6iauI9aqib5SFv4TiaGHeWicZNnV8FHicp63wfU96rGZ9iaEwHIEOUYjbVkYKib27G00UGg7TeIKP5gcK2yEU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

图3 文本Transformer与图像Transformer对比

上图为GPT类模型：对句子做tokenization，掩码自注意力处理序列，取最后上下文向量预测下一个词； 下图为Vision Transformer：图像切分为图像块token，无掩码全局自注意力，依靠class token完成图像分类。

BERT是另一个可与ViT对照的模型：BERT不预测下一词，而是通过掩码语言建模恢复句子中被遮挡的token，因此采用无掩码双向自注意力捕获全局上下文。Vision Transformer沿用BERT这种仅编码器架构，将其应用于图像块序列并新增class token，在图像领域实现了类似BERT的全局序列理解能力。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibtSc0tiaOPnlADd1zQhZNjy2XiaquNw4xg2cX1vr2G8sDL5God53QPgjicckZagyhWHGicUUoib3z9EMqcsbN4Pt9y1je5ZdxozEmc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

图4 BERT掩码语言建模流程

模型输入带有掩码符号的句子，通过无掩码自注意力处理完整序列，在输出端还原被掩码token的原始文本。

完成文本与图像Transformer的宏观对比后，我们聚焦Vision Transformer处理图像的完整流程。下一节将详解图像分割为小块、转为向量、整理为一维序列（与文本token序列形式一致）的分块嵌入流程，这也是标准Transformer编码器能够直接处理图像的前置步骤。

## 1.2 将Transformer适配图像：分块嵌入与展平

要让Transformer处理图像，首要问题是：如何将二维图像转换为文本Transformer可接收的一维token序列？Vision Transformer的解决方案是将图像划分为固定尺寸的网格小块，每个小块视作一个token。本节以640×640像素猫咪图像为例拆解分块嵌入流程，并介绍两种主流实现方案：展平+线性投影、等效卷积层实现。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibfAPILTjcIVRn4skKlh2AD0cJ26bEssIH3ggsM3MqeKOWHYv8MYXmj2qDQkVUygCIyFadcuxBE77TDymwUmFwFsMl5zRfuSQU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

图5 640×640图像分块操作

图像被分割为4×4网格、互不重叠的160×160像素小块，编号1~16。这16个小块将作为Transformer的16个token，后续会在序列最前端拼接单独的class token，最终总序列长度为17。

假设图像高为、宽为、单块尺寸为，则图像块总数计算公式：

前提是、均可被整除。若为边长的正方形图像，公式简化为：

以本例640×640图像、块尺寸代入计算：

图像分割完成后，每个小块需要映射至统一维度的嵌入向量空间，与文本模型中词嵌入逻辑一致。原始ViT论文采用的最简方案为：将每个小块展平，再通过线性层做投影。单块图像维度为（为通道数），展平后向量长度为。

所有图像块按照固定顺序排列为一维序列，而非二维网格：例如从左上角小块开始，逐行从左至右遍历，直至右下角。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZ9L7iaUyrKr3tVXBj61pLsia51Lib9Ht0u5uvCZnOyX9flAuc87O10ltRj07c4Ex7ibyeXpYzbxk2gC3wnYUicibJ9V0Akr0Xsics191c/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

图6 16个图像块整理为一维序列

每个小方格仍是原始图像块，但在Transformer视角下，它们会被整理成长度16的token序列。

### 无卷积实现分块嵌入

本小节不使用任何卷积层搭建分块嵌入模块。我们将160×160×3的图像小块视作独立小图，把全部像素展平为长向量，再输入可学习线性层，输出维度为的图像块嵌入向量。该逻辑与语言模型词嵌入完全对应：每个图像块等同于独立token，嵌入参数由数据端到端学习得到。

回到前文640×640猫咪图像示例，图像被划分为4×4互不重叠、单块160×160像素的网格，共16个小块，每一块对应一个token。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibwad7Hn7hg97OveiaovJ9N2jlK1X1Y7EV6GY6mQ9nDQ4uTpia3ic0hoRhShtQV2qicCVXv7eVSXAEvK7icZUViaBzHWNK5XP7Ff9r4s/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

图7 640×640 RGB图像中第10块（猫咪眼部区域）

单块尺寸，拆分红、绿、蓝三个通道，展示若干像素的(R,G,B)数值。所有像素数值会被展平为一维长向量，用于后续生成图像块嵌入。

以第10块（猫咪眼部区域）为例，RGB小块三维张量形状为：

其中为色彩通道数。图7将该小块拆分三通道，并展示若干像素的(R,G,B)三元组。无卷积分块嵌入的第一步，是将张量按固定顺序拼接所有通道、所有像素值，完成展平。记第个图像块展平后的向量为，其长度为：

对于160×160 RGB图像块：

即每个小块展平后包含76800个数值。展平仅为张量重塑操作，无任何可学习参数。

但Transformer无法直接使用长度76800的原始像素向量，它需要维度更小的嵌入向量（示例示意图取，标准ViT-Base模型取）。最简单的映射方式是对所有展平后的图像块共享同一个线性层，引入可学习权重矩阵与偏置向量，定义第个图像块的嵌入：

其中：

维度匹配逻辑：将长度的向量压缩映射为长度的向量，提供偏移量。以示例为例，形状为，长度为32。输出：

即维**图像块嵌入向量**。

与为全局共享参数，对所有图像、所有图像块复用，与模型其余权重一同端到端训练。可以将的每一行理解为一组可学习模板，读取整块图像信息后输出单个数值；堆叠组输出结果，即得到完整嵌入向量。

至此，单个图像块已转换为独立token。对全部个图像块重复展平+线性投影操作，得到一组图像块嵌入：

按照固定、确定的顺序（逐行从左至右）排列所有向量，拼接为矩阵：

该矩阵完全等价于文本Transformer中的token嵌入矩阵，唯一区别是矩阵每一行编码一整块160×160彩色图像区域，而非单词/子词。

完整流程总结：嵌入前模型读取个原始图像块，每个形状；展平后得到条长度的向量；经过线性投影后输出条维图像块嵌入。下一节加入专用class token与位置嵌入后，矩阵尺寸扩展为，作为Transformer编码器的输入。

### 卷积实现分块嵌入

本节介绍如何仅用一层Conv2D完成与上述方案完全等价的操作：将卷积核尺寸、步幅与图像块尺寸设为相等，卷积层直接把640×640图像输出为特征图，通道维度恰好等于嵌入维度（示例取32通道）。特征图每个空间位置对应一个图像块token。

核心思路：卷积核尺寸、步幅，每个图像块仅被卷积遍历一次，压缩为特征向量，输出网格尺寸为。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZ9nKKnQPw2xDqURBuTerNOQjeA1ydHmQS3w7ibUtIWMduiauZ083oicgiccuXiaumgrRd3aAmnWuj3wfhJjaFhKeKiaLumj1GTyGHFkc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

图8 卷积实现分块嵌入流程

输入为640×640三通道图像，使用32个尺寸的可学习卷积核，步幅160。输出为4×4网格、32通道特征图，网格内每个单元格对应原图一个160×160图像块，32个通道即为维图像块嵌入。

沿用前文示例，卷积核空间尺寸160×160、步幅160，输入张量形状，通道数；输出通道数等于目标嵌入维度（示例）。卷积核尺寸、步幅均设为160，卷积窗口以无重叠160像素步长遍历图像，零填充保证图像完整划分为图像块，无额外边界像素。

卷积层包含个独立卷积核，每个核权重张量形状。卷积处理图像时，每个核以160像素步长滑动，每个图像块输出一个特征值。由于步幅等于核尺寸，相邻感受野无重叠。卷积输出张量形状为：

网格内每个空间坐标对应原图一个图像块，该位置32维向量由32个卷积核共同输出，作用等同于前文线性矩阵的各行权重。将4×4空间网格展平为长度16的一维序列，读取每个位置的32维向量，最终得到与无卷积方案完全一致的图像块嵌入。

两种实现方案证明：Vision Transformer不依赖特定分块提取方式，核心目标是生成条维图像块嵌入向量。选择展平+线性层还是定制卷积层仅为工程实现差异；只要得到尺寸的图像块token矩阵，后续ViT计算流程完全统一。

### 拼接class token、构建完整序列

目前我们得到条维图像块嵌入。针对分类任务，Vision Transformer引入额外token——**class token**。该token不对应任何图像块，是独立可学习向量，拼接在序列最前端，通过自注意力聚合所有其他token的全局信息。

记class token嵌入向量：

该向量为可训练参数，模型初始化时随机赋值，与其余权重同步优化。将与图像块嵌入拼接，得到完整序列矩阵：

送入Transformer编码器的总token数量：

以640×640图像、块尺寸160为例，，叠加1个class token，总token数量17，每条向量维度。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZicJruackfkPh5Ac10EKGWvpvkHk266ofEP4ZXUfPydjXDgEUEgmF2ibAspt0UhJThXicxUk0xxj2Nu8FaQBnL8ddE003QMN0qvP4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

图9 完整token序列构建流程

16个图像块经共享线性投影/卷积转换为32维嵌入；前端追加可学习class嵌入，输出矩阵；叠加位置信息后，该矩阵作为Vision Transformer编码器输入。

## 1.3 Vision Transformer中的位置编码

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibMbFlY7gm3XVjq1hWbJmnrI7tGwVhnQLojeA5I59PicSgS5EDRmBHN6ZFqmn10TOZGlWQ4RuXzrNCs6puO06HjoZKLdUjG1pKE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

图10 有无位置嵌入对比

左图：无位置嵌入时，16个猫咪图像块可任意打乱输入模型，图像空间结构完全丢失；

右图：添加位置嵌入后，每个token绑定其在网格中的原始坐标，模型能够识别图像块空间位置。

ViT自注意力机制本身不具备序列顺序感知能力。如图10左图，若交换猫咪耳朵图像块与纯色背景块的顺序，编码器会正常处理打乱后的序列，无法察觉空间错乱。但图像具备严格空间逻辑：猫咪眼部图像块与纯紫色背景块语义完全不同。为让模型识别每个token在原图网格中的坐标，需在token送入编码器前，为每个向量叠加位置嵌入。

完成图像块嵌入、拼接class token后，我们得到长度、维度的token序列，存储于矩阵：为class token，为图像块嵌入。Vision Transformer引入可学习位置嵌入矩阵：

矩阵每行为可学习向量，代表第个token的位置信息，训练过程中与其余参数同步更新。

批次大小为时，同一份位置矩阵广播至整个批次，逐元素相加得到编码器最终输入：

送入Transformer编码器的张量尺寸为。序列长度、嵌入维度均无变化，但每个token同时承载两类信息：图像块视觉内容、该块在原图网格中的空间坐标。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ9TCZAfWONp3e2692rYS5H0bQibliclzKoY8MmU43VzIkJs0fG3SjovCLJ0LfcEBxcNoTsTCBG7jrFw7V3icGQhauJVeQzu3jlh6A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

图11 ViT可学习位置嵌入原理

每个图像块嵌入与class token 分别匹配一条可学习位置向量；相加后生成新token ，同时编码图像块内容与空间位置。序列长度仍为，嵌入维度，模型可获取完整图像空间结构信息。

## 1.4 面向分类任务的仅编码器架构

Vision Transformer仅保留原始Transformer架构中的编码器模块。图像转换为token序列后，送入层堆叠、结构完全相同的编码器块；每个块包含多头自注意力、前馈MLP、残差连接与层归一化，无用于预测未来token的解码器。架构核心设计为：序列前端拼接独立可学习class token，编码器作为全局特征提取器；最后一层编码器输出后，仅取出class token的最终隐向量，送入小型MLP头输出分类logit。从该角度看，ViT是面向分类任务、仅使用编码器的模型。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ9Dbm6nicRqiaicSkic4OiccsicV7uXQB0jz0wib6lgrJWcuJhQBj4CrDHw5vvoErJP2ZdTWFeGbhJTEvBFn27yXibdwyYsXh7nazLuDaA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

图12 仅编码器结构Vision Transformer完整流程

叠加位置信息的图像块token序列，经过层堆叠编码器块；class token最终上下文向量作为MLP分类头的输入。

### 从分块嵌入到上下文向量完整链路

前文已得到融合图像块信息、位置信息的token矩阵，叠加class token后序列总长度，单token嵌入维度。所有token向量按行拼接得到矩阵：

以示例、为例，矩阵尺寸，作为Transformer编码器栈的输入。在编码器视角下，该矩阵与语言模型的token嵌入矩阵完全一致：一批次多条序列，每条序列长度17，每个token用32维向量表示。

编码器不会改变序列长度，经过任意一层编码器块后，输出矩阵尺寸仍为；但每一行向量通过自注意力与MLP融合了其余所有token的信息，更新后的向量称为上下文向量——同时编码自身token内容、其他token提供的全局上下文。

### Transformer编码器与注意力机制

拆解单层编码器块内部，先聚焦自注意力子层。块输入矩阵记为：

代表编码器块在堆叠结构中的深度，矩阵每行存储当前时刻单个token的特征表示。自注意力机制将该矩阵转换为尺寸完全相同的新矩阵：每个token读取全部其他token的特征，并计算各token的关注度权重。

第一步：将token特征投影至三组独立空间——查询、键、值。将分别与三组可学习权重矩阵相乘：

为单头注意力的头维度。三组权重矩阵对序列所有位置共享，训练过程中学习更新，计算得到三组新矩阵：

三组矩阵尺寸均为。直观理解：token  的查询向量代表该token希望从全局上下文检索的信息；键向量代表该token对外提供的特征信息；值向量代表当其他token关注本token时，会融合的实际特征数据。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZibnia4vcImuAu9WNsNUyyO5okeO6OMgqvjP2llQhPsttcN0DIoSjkISKHqRJKmzShFjaXTNau4B3jYyreH2d2tvZCcvtiavyHIO0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

图13 单头注意力内部流程

图像块嵌入序列经线性投影生成三组并行向量：键、查询、值；、、为可学习全局共享权重，决定该注意力头检索、输出的特征类型。

第二步：查询与键计算相似度，生成注意力权重。对任意查询向量，与所有键做缩放点积，得到每一组位置对应的相似度标量得分：

沿维度做softmax归一化，将得分转换为概率分布：

满足约束：

且

系数含义：**token  对 token  的关注权重**。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ9g7WWus4E3jcypWLxT7pBCRaexH4duTUOFaW1yHOuiaSPAkFFugHiaBsiaUEytj3Bjz5RJhyAf5vZpABKqneEdaG9BIicbibJZ59KQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)

图14 单查询缩放点积注意力计算流程

单个图像块对应的查询向量与全部键向量计算相关性得分，softmax归一化得到注意力权重；权重数值代表当前图像块需要对其余每个图像块分配多大关注度。

第三步：基于注意力权重加权融合值向量。对token ，将全部值向量按权重求和：

输出向量：

即该注意力头输出的token  新特征表示，融合了所有token的值向量；与当前token相关性越高的位置，加权系数越大。举例：若某注意力头学习聚焦猫咪眼部区域，计算class token上下文向量时，眼部周边图像块的值向量会获得更大权重。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZ9HtErqRAmJQsePldW2TNllickZb7TXA2oDNHYSU3zQXYy94Caicn3ibNMHRzicDPajKrzG0RQPchZchxTUiaoJkLrJYKwx4KasxnmQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

图15 注意力权重矩阵生成上下文矩阵

尺寸的注意力权重矩阵与堆叠值向量相乘，生成全新token特征矩阵；输出矩阵每一行，是单个查询位置对全部值向量的加权求和结果。

实际ViT均采用多头注意力，而非单头注意力：并行执行上述流程，每组头配备独立投影矩阵、、。每个头拥有独立头维度，所有头输出拼接后，通过另一组可学习投影还原原始嵌入维度，得到注意力子层输出矩阵（尺寸）；该矩阵送入MLP子层、叠加残差连接，生成更新后的特征矩阵。

### 上下文向量与输出维度演变

引入标准化符号跟踪特征经过编码器栈的变化： 令，代表叠加图像块嵌入、位置嵌入后的初始token矩阵。经过第层编码器块后，输出记为：

其中代表第层、第个token的上下文向量。索引对应class token，对应图像块token。编码器栈全程不改变序列长度，所有尺寸恒为。以示例、为例，每层编码器输入、输出矩阵均为。

经过最后一层编码器，得到。矩阵中最重要的向量是，即class token最终上下文向量。训练阶段，该向量通过多层自注意力、MLP反复聚合所有图像块token信息，成为整幅图像的紧凑全局特征摘要。将送入小型MLP头，映射为各类别logit（例如ImageNet-1k数据集输出1000维）；logit经过softmax归一化，输出各类别预测概率分布。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZicQQ2wz2DPBoEzPBLE8L66E5L9vOM95DQEBy7ELgtYtg0x0LLAL3lhsaDa597rWia6VxNUn9QDa9qic7ud0YYyl9yd6f64QbQj9M/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)

图12 编码器栈上下文向量变化

每层编码器块接收矩阵，输出同尺寸矩阵；class token最终上下文向量是整幅图像的学习摘要，仅该向量送入分类头。

### MLP分类头与图像分类逻辑

序列完整经过编码器栈后，核心特征提取流程全部完成。回到猫咪图像示例：生成个图像块token，前端拼接可学习class token，全部映射至维嵌入空间；经过层编码器块，输出最终序列矩阵：

矩阵每行为第个token的上下文向量，对应class token，对应图像块。示例中、，尺寸。

图像分类任务不将全部17条上下文向量送入分类网络，遵循原始ViT设计：仅使用class token最终上下文向量：

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ8qEZfHzflCWiaIUTqWJu5rQnQSjKlZogacDKlbF7mkZiaeBicKjbRD6SbxwvAywcw8fHBnFLDsFSaNFDMN5gibyzVAPqDic5JSN8dY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)

图16 Vision Transformer分类器完整架构

图像转换为带位置嵌入的图像块token，经过多层Transformer编码器；最终仅读取专用class token的上下文向量，送入MLP头输出分类预测。

该向量在每一层编码器均与所有图像块token建立注意力关联，天然成为整幅图像的学习摘要。仅使用单条摘要向量可简化架构、大幅降低分类头参数量。理论上也可拼接/池化全部图像块上下文向量，但会大幅提升分类输入维度，且ViT原始实验未观测到显著精度提升。

MLP头是标准前馈分类网络，输入，输出每个类别对应的logit。最简实现为单层线性层，权重矩阵、偏置，为类别总数（猫、狗、鸟类等），logit向量计算：

工业界主流ViT实现会采用双层小型MLP替代单层线性层：先将投影至隐藏维度，接入GELU非线性激活，可选加入dropout正则化，最后映射至维类别空间。双层结构为分类器提供更强表征重塑能力，优化编码器输出特征至分类得分的映射效果。

输出是未归一化得分（logit），每个数值对应一个类别。推理阶段取最大logit索引作为预测标签；训练阶段将输入softmax得到类别概率分布，与真实标签计算交叉熵损失。损失梯度反向传播至MLP分类头，再传导至Transformer编码器、图像块嵌入与位置嵌入层，实现Vision Transformer全链路端到端训练。

## 1.5 ViT的优势与缺陷

Vision Transformer将图像建模为token序列，完全依靠自注意力捕捉图像区域关联，提供一套逻辑简洁、拓展灵活的CNN替代方案。核心优势为全局上下文建模：从第一层编码器开始，每个图像块可直接关注图像内所有其他块，对远距离依赖建模能力极强，例如物体相隔较远部件的关联、前景与背景区域的交互。同时ViT架构具备极强的数据、模型规模拓展性：在大规模数据集上训练时，Vision Transformer精度常超越卷积网络，证明充足数据下卷积带来的局部归纳偏置并非必需。架构简洁是另一大优势：除分块嵌入模块外，模型主体与语言领域标准Transformer编码器完全对齐，视觉、语言任务的算法思路、优化方案、工具链可通用复用。

但上述优势伴随显著取舍：Vision Transformer在中小规模数据集上的数据利用效率普遍弱于卷积网络。卷积天然自带局部性、平移等变性归纳偏置，而ViT需要从数据中从零学习全部视觉规律，数据量不足时性能会明显下滑。自注意力机制计算、内存开销更高，计算复杂度随图像块数量呈二次增长；高分辨率图像场景下极易成为性能瓶颈。因此大量实用ViT变体引入分层结构、窗口注意力、CNN-Transformer混合架构缓解该问题。总结：数据、算力充足时ViT表现优异；资源受限场景下，需要针对性架构优化才能维持竞争力。

## 1.6 Vision Transformer实际落地应用

Vision Transformer现已广泛应用于各类视觉落地任务，尤其适合拥有大规模数据集、支持预训练的场景。图像分类领域，ViT及其变体成为深度卷积网络的强力替代方案：在大规模图像数据集预训练后，下游任务微调可取得SOTA精度。除分类外，ViT在目标检测、图像分割任务中效果突出，全局上下文建模能力对两类任务增益显著：复杂场景小目标检测、大范围空间连续结构分割，均需要远距离图像块直接关联。

工业落地场景中，ViT大量用于医学影像、遥感、自动驾驶系统。医学影像领域，ViT建模高分辨率扫描图（MRI、病理切片）内复杂空间关联，长距离依赖特征对疾病诊断至关重要；卫星、航拍遥感图像中，ViT用于土地利用分类、地物变化检测、大范围场景理解。同时ViT是现代多模态系统核心组件：图像需与文本、音频等模态对齐，图像-文本编码器均以ViT作为视觉骨干，输出可与语言Transformer兼容的视觉特征。因此ViT成为图像描述、视觉问答、大型多模态模型的基础模块，实现跨感知任务的统一架构。

## 1.7 实操环节：微调ViT完成图像分类

微调Vision Transformer完整代码仓库链接：

https://github.com/VizuaraAI/Transformers-for-vision-BOOK

本节基于真实高分辨率图像分类任务，微调预训练Vision Transformer。Oxford-IIIT Pet数据集具备丰富精细视觉特征，匹配ViT的归纳偏置，是演示ViT迁移学习的理想数据集。我们采用ImageNet预训练ViT-Base模型，微调完成宠物图像品种分类，遵循现代视觉系统标准迁移学习流程。

### 数据集与任务设定

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/Kltic3d4ibvZibdz2XwLT1t1qs4ZfVwiatBNyaprO55eoCxnP01uSmWGysMUk2BkJnHU6ZDlkOm9uldtC8NCO1D5ArGe43v5FFPDs948KfHqicfg/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=18)

Oxford-IIIT Pet数据集包含7349张猫狗图像，共37个细粒度品种类别；图像分辨率普遍大于200×200像素，包含丰富纹理、外形、空间特征，单张图像仅对应一个品种标签，属于多分类任务。尽管原图分辨率较高，预训练ViT统一要求输入尺寸224×224，预处理阶段统一缩放至该分辨率。

该数据集适合演示ViT微调的三大原因：

1. 细粒度分类：区分37个宠物品种，需要模型捕捉毛发纹理、耳朵形状、身体比例等细微视觉差异，恰好是自注意力擅长的远距离空间推理任务；
2. 视觉复杂度充足：图像包含自然背景、多变姿态、复杂光照，提供真实迁移学习挑战；
3. 数据规模适中：训练集约3680张、测试集约3669张，单GPU可快速完成微调，同时数据量足以产出有效训练结果。

### 安装依赖、定义固定超参

编写模型代码前，先安装所需库并定义全程固定的超参数：

#### 代码清单1.1 安装依赖、定义常量

```
!pip install torchmetrics -q
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import get_cosine_schedule_with_warmup

from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm.auto import tqdm
```

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

NUM_CLASSES = 37
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
```

固定随机种子保证实验可复现；`NUM_CLASSES = 37`匹配Oxford-IIIT数据集37个宠物品种；输入尺寸224匹配预训练ViT-Base标准输入分辨率。

### 数据集加载与可视化探索

使用`torchvision.datasets.OxfordIIITPet`下载、加载数据集；数据集划分train-val划分用于训练，独立test划分用于验证：

#### 代码清单1.2 加载Oxford-IIIT Pet数据集

```
raw_train = datasets.OxfordIIITPet(
 root="./data",
 split="trainval",
 target_types="category",
 download=True
)

class_names = raw_train.classes
print(len(class_names))
print(class_names[:10])
```

打印输出：

```
37
['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue']
```

#### 代码清单1.3 可视化数据集样本图像

```
plt.figure(figsize=(12, 6))

for i in range(8):
 img, label = raw_train[i]
 plt.subplot(2, 4, i + 1)
 plt.imshow(img)
 plt.title(class_names[label])
 plt.axis("off")

plt.show()
```

运行后输出样本图像网格。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZibiaEeSNgwoE2OpbeJUJ1hcDsQUEBIZa5QQ0pJ6WIVua268ffe4FQblDwMyQBpBJGer9ZaxNO9icEBgbazRkzAsmSbXs4gZnqpzo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)

从可视化结果可见：图像纹理丰富、宠物姿态多变、背景复杂，这类场景下图像块间远距离注意力机制收益极高。

### 预处理流水线与数据加载器构建

预训练Vision Transformer对预训练阶段使用的归一化均值、标准差高度敏感。我们加载`ViTImageProcessor`提取标准均值、方差，分别构建训练、验证预处理流水线：

#### 代码清单1.4 基于ViTImageProcessor搭建预处理流水线

```
processor = ViTImageProcessor.from_pretrained(
"google/vit-base-patch16-224"
)
print(processor)
```

输出：

```
ViTImageProcessor {
 "do_convert_rgb": null,
"do_normalize": true,
"do_rescale": true,
"do_resize": true,
"image_mean": [
 0.5,
 0.5,
 0.5
 ],
"image_processor_type": "ViTImageProcessor",
"image_std": [
 0.5,
 0.5,
 0.5
 ],
"resample": 2,
"rescale_factor": 0.00392156862745098,
"size": {
"height": 224,
"width": 224
 }
}
```

```
image_mean = processor.image_mean
image_std = processor.image_std

train_transforms = transforms.Compose([
 transforms.RandomResizedCrop(IMAGE_SIZE),
 transforms.RandomHorizontalFlip(),
 transforms.ToTensor(),
 transforms.Normalize(mean=image_mean, std=image_std)
])

val_transforms = transforms.Compose([
 transforms.Resize(IMAGE_SIZE),
 transforms.CenterCrop(IMAGE_SIZE),
 transforms.ToTensor(),
 transforms.Normalize(mean=image_mean, std=image_std)
])
```

训练流水线加入随机缩放裁剪、水平翻转做数据增强；验证流水线使用固定缩放+中心裁剪，保证评估可复现。两套流水线均采用ViT预训练配套的ImageNet归一化参数。

预处理完成后，单张图像张量尺寸统一为，匹配ViT输入规范。

构建训练、验证数据集，封装为PyTorch DataLoader：

#### 代码清单1.5 构建训练、验证数据加载器

```
train_dataset = datasets.OxfordIIITPet(
 root="./data",
 split="trainval",
 target_types="category",
 transform=train_transforms
)

val_dataset = datasets.OxfordIIITPet(
 root="./data",
 split="test",
 target_types="category",
 transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

### 加载预训练模型

通过Hugging Face transformers库加载ImageNet预训练ViT-Base模型；参数`num_labels=NUM_CLASSES`会自动替换原始1000类ImageNet分类头，生成适配37个宠物品种的全新线性输出层；`ignore_mismatched_sizes=True`屏蔽分类头尺寸不匹配的警告：

#### 代码清单1.6 加载预训练ViT-Base并替换分类头

```
model = ViTForImageClassification.from_pretrained(
 "google/vit-base-patch16-224",
 num_labels=NUM_CLASSES,
 ignore_mismatched_sizes=True
).to(device)
```

```
print(model)
```

模型结构输出：

```
ViTForImageClassification(
 (vit): ViTModel(
 (embeddings): ViTEmbeddings(
 (patch_embeddings): ViTPatchEmbeddings(
 (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
 )
 (dropout): Dropout(p=0.0, inplace=False)
 )
 (encoder): ViTEncoder(
 (layer): ModuleList(
 (0-11): 12 x ViTLayer(
 (attention): ViTAttention(
 (attention): ViTSelfAttention(
 (query): Linear(in_features=768, out_features=768, bias=True)
 (key): Linear(in_features=768, out_features=768, bias=True)
 (value): Linear(in_features=768, out_features=768, bias=True)
 )
 (output): ViTSelfOutput(
 (dense): Linear(in_features=768, out_features=768, bias=True)
 (dropout): Dropout(p=0.0, inplace=False)
 )
 )
 (intermediate): ViTIntermediate(
 (dense): Linear(in_features=768, out_features=3072, bias=True)
 (intermediate_act_fn): GELUActivation()
 )
 (output): ViTOutput(
 (dense): Linear(in_features=3072, out_features=768, bias=True)
 (dropout): Dropout(p=0.0, inplace=False)
 )
 (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
 (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
 )
 )
 )
 (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
 )
 (classifier): Linear(in_features=768, out_features=37, bias=True)
)
```

### 冻结骨干网络，仅训练分类头

迁移学习主流策略：冻结全部预训练骨干参数，仅训练随机初始化的全新分类头。该方案训练速度快、显存占用低，且预训练视觉特征足以支撑下游任务取得良好效果：

#### 代码清单1.7 冻结骨干、统计可训练参数量

```
for param in model.parameters():
 param.requires_grad = False
for name, param in model.named_parameters():
 if "classifier" in name:
 param.requires_grad = True
```

编写函数验证冻结效果，统计参数总量：

```
def print_model_parameters(model):
 trainable_params = 0
 frozen_params = 0
 all_param = 0

for _, param in model.named_parameters():
 num_params = param.numel()
 all_param += num_params

if param.requires_grad:
 trainable_params += num_params
else:
 frozen_params += num_params

print(f"trainable params: {trainable_params:,}")
print(f"frozen params: {frozen_params:,}")
print(f"all params: {all_param:,}")
print(f"trainable%: {100 * trainable_params / all_param:.2f}%")

# 执行函数
print_model_parameters(model)
```

输出结果：仅分类头可训练，总参数量8600万左右，可训练参数仅2.8万：

```
trainable params: 28,453
frozen params: 85,798,656
all params: 85,827,109
trainable%: 0.03%
```

### 训练前推理校验

正式微调前，在单张验证图像上跑推理，确认基线效果：分类头随机初始化，预测结果应为完全随机、置信度极低：

#### 代码清单1.8 训练前单图推理校验

```
model.eval()

image, label = val_dataset[0]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
 outputs = model(pixel_values=image)
 logits = outputs.logits
 probs = torch.softmax(logits, dim=1)
 pred = probs.argmax(dim=1).item()
 confidence = probs.max(dim=1).values.item()

print(
 f"[训练前推理校验]\n"
 f" 真实品种 : {class_names[label]}\n"
 f" 预测品种 : {class_names[pred]}\n"
 f" 预测置信度 : {confidence:.2f}"
)
```

输出：

```
[训练前推理校验]
 真实品种 : Abyssinian
 预测品种 : Shiba Inu
 预测置信度 : 0.05
```

模型预测错误、置信度极低，证明分类头需要训练优化。

### 优化器、学习率调度、损失函数配置

采用AdamW优化器，学习率，搭配带线性预热的余弦学习率调度；预热阶段稳定随机初始化分类头的早期训练：

#### 代码清单1.9 配置AdamW、余弦调度器、损失函数

```
optimizer = optim.AdamW(
 filter(lambda p: p.requires_grad, model.parameters()),
 lr=3e-4,
 weight_decay=1e-4
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
 optimizer,
 num_warmup_steps=warmup_steps,
 num_training_steps=total_steps
)

criterion = nn.CrossEntropyLoss()
accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
```

#### AdamW微调说明

AdamW是Adam优化器变体，将权重衰减与梯度更新解耦，避免正则项干扰自适应学习率更新，泛化效果更优，是Transformer预训练、微调的标准优化器。

### 主训练循环

标准PyTorch训练流程：逐轮遍历小批次，计算交叉熵损失、反向传播、更新分类头权重；每轮训练结束后在验证集评估精度：

#### 代码清单1.10 训练+验证主循环

```
train_losses, val_accuracies = [], []

for epoch in range(EPOCHS):
 model.train()
 running_loss = 0.0

for imgs, labels in tqdm(train_loader):
 imgs, labels = imgs.to(device), labels.to(device)

 outputs = model(pixel_values=imgs)
 loss = criterion(outputs.logits, labels)

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 scheduler.step()

 running_loss += loss.item()

 train_losses.append(running_loss / len(train_loader))

 model.eval()
 accuracy.reset()
 with torch.no_grad():
for imgs, labels in val_loader:
 imgs, labels = imgs.to(device), labels.to(device)
 preds = model(pixel_values=imgs).logits.argmax(dim=1)
 accuracy.update(preds, labels)

 val_accuracies.append(accuracy.compute().item())

print(f"Epoch {epoch+1}: "
 f"Loss={train_losses[-1]:.4f}, "
 f"Val Acc={val_accuracies[-1]:.4f}")
```

关键细节：每轮训练开头执行`model.train()`开启dropout；验证前执行`model.eval()`关闭正则层；`scheduler.step()`在每个参数更新步骤后执行，符合余弦预热调度标准逻辑。

### 绘制训练曲线

训练完成后绘制训练损失、验证精度曲线，判断模型收敛状态：

#### 代码清单1.11 绘制损失与精度曲线

```
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("训练损失")
plt.xlabel("轮次")
plt.ylabel("损失值")

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title("验证集精度")

plt.xlabel("轮次")
plt.ylabel("精度")
plt.tight_layout()
plt.show()
```

健康训练曲线特征：训练损失持续下降，验证精度前多轮稳步上升后趋于平稳；仅训练分类头的收敛速度通常很快。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibQYPg3KdXEic0icW9dJU9Cpe7dV0eIMQ8yTjQNV9rbsr3YoP2vgWAtXIE055BwTkiaxY3hmhH6TarYKvDfM356xRNHt2IQQrQiap8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)

### 训练后推理校验

验证微调完成的模型预测效果，在验证集单张图像测试：

#### 代码清单1.12 训练后单图推理

```
model.eval()

image, label = val_dataset[0]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
 logits = model(pixel_values=image).logits
 pred = logits.argmax(dim=1).item()

print("训练完成预测 → 预测品种：", class_names[pred],
 "| 真实品种：", class_names[label])
```

输出示例：

```
训练完成预测 → 预测品种：Abyssinian | 真实品种：Abyssinian
```

### 混淆矩阵完整评估

混淆矩阵可直观定位模型识别缺陷，展示易混淆品种配对：

#### 代码清单1.13 计算并展示混淆矩阵

```
all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
for imgs, labels in val_loader:
 imgs = imgs.to(device)
# 获取预测结果
 preds = model(pixel_values=imgs).logits.argmax(dim=1)

 all_preds.extend(preds.cpu().numpy())
 all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(30, 30))

disp.plot(ax=ax, xticks_rotation=45, colorbar=True)

plt.tight_layout()

plt.show()
```

对角线数值代表正确预测样本；非对角线数值代表类别混淆样本。外观特征相似的品种（例如不同短毛猫）非对角线数值会显著更高。该细粒度分析可指导后续优化：扩充数据集、增强数据增广、更换更大规模模型。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZ8uwO9tvXYwYheqibgu6hR7S6az9Qic2d2sZIMtUicLnVCoZ6SHoZpU5VrvBMr0YeIVKNg16dfKx2MEueDic6AcxZnPxMZG7icO39x0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)

### 保存微调完成的模型

存储微调权重，后续可直接加载推理、无需重复训练：

#### 代码清单1.14 保存模型权重

```
torch.save(model.state_dict(), "vit_finetuned_final.pth")
print("模型保存成功。")
```

`.pth`文件仅存储模型`state_dict`（层名-参数张量映射字典）。重新加载时，新建同配置`ViTForImageClassification`实例，调用`model.load_state_dict(torch.load("vit_finetuned_final.pth"))`即可恢复权重。

## 参考资料

### 原始论文

An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## 拓展阅读

Dr Sreedath Panat

Vision Transformer Paper Dissection

Build Vision Transformer from Scratch

## 1.8 本章总结

1. Vision Transformer将Transformer架构适配图像任务：把固定尺寸图像小块视作token，第一层即可实现全局自注意力。卷积网络逐层堆叠缓慢扩大感受野，而ViT可直接关联图像内任意两个区域。
2. 分块嵌入将二维图像转换为一维token序列，两种等价实现：图像块展平后线性投影、卷积核尺寸/步幅等于图像块尺寸的单层卷积。
3. 序列前端拼接独立可学习class token，通过自注意力聚合全部图像块全局信息；最后一层编码器输出后，仅使用class token上下文向量作为整图紧凑摘要，送入MLP分类头输出预测。
4. 可学习位置嵌入叠加至每个token，让模型识别每个图像块在原图网格中的空间坐标。
5. ViT采用仅编码器架构：完整图像块序列经过层堆叠编码器块（多头自注意力+前馈网络）；每一层不改变序列长度、嵌入维度，逐层迭代优化token上下文表征。
6. Vision Transformer在大规模数据、大模型规模下性能优异，但中小数据集数据利用效率低于CNN；各类实用变体通过分层结构、窗口注意力缓解自注意力二次计算开销。
7. 预训练ViT下游分类微调标准迁移学习pipeline：冻结预训练骨干网络、替换全新分类头，仅训练分类头；采用带预热的余弦学习率调度优化训练过程。
