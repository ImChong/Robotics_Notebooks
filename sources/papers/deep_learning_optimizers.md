# deep_learning_optimizers

> 来源归档（ingest）

- **标题：** Deep Learning Optimizers — SGD / Momentum / Nesterov / Adagrad / RMSProp / Adadelta / Adam / AdamW / Lion
- **类型：** paper + course + framework docs
- **来源：** 经典优化论文、深度学习教材、PyTorch 官方文档
- **入库日期：** 2026-06-27
- **一句话说明：** 覆盖神经网络训练中最常用的 9 类一阶优化器及其一手出处，支撑各优化器独立 method 页与选型对比页。

## 核心论文摘录（MVP）

### 1) A Stochastic Approximation Method (Robbins & Monro, 1951)
- **链接：** <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full>
- **核心贡献：** 提出随机逼近框架：用含噪梯度估计的迭代更新替代精确梯度，奠定 **SGD** 的理论起点。
- **关键更新：** $\theta_{t+1} = \theta_t - \eta_t g_t$，$g_t$ 为随机梯度。
- **对 wiki 的映射：**
  - [SGD](../../wiki/methods/sgd.md)
  - [Deep Learning Optimizers 对比](../../wiki/comparisons/deep-learning-optimizers.md)

### 2) Some Methods of Speeding Up the Convergence of Iteration Methods (Polyak, 1964)
- **链接：** <https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=zvmmf&paperid=4607>
- **核心贡献：** **Heavy-ball / Momentum** 方法：在梯度方向外叠加历史速度项，加速凸优化收敛。
- **关键更新：** $v_{t+1} = \mu v_t - \eta \nabla f(\theta_t)$，$\theta_{t+1} = \theta_t + v_{t+1}$。
- **对 wiki 的映射：**
  - [SGD Momentum](../../wiki/methods/sgd-momentum.md)

### 3) A Method for Solving the Convex Programming Problem with Convergence Rate O(1/k²) (Nesterov, 1983)
- **链接：** <https://doi.org/10.1016/0273-0979(83)90028-3>（英译综述）；原始俄文见 Soviet Math. Dokl.
- **核心贡献：** **Nesterov 加速梯度（NAG）**：在动量更新前对参数做「前瞻」梯度评估，凸情形达到 $O(1/k^2)$ 收敛率。
- **对 wiki 的映射：**
  - [Nesterov Momentum](../../wiki/methods/nesterov-momentum.md)

### 4) On the Importance of Initialization and Momentum in Deep Learning (Sutskever et al., 2013)
- **链接：** <https://proceedings.mlr.press/v28/sutskever13.html>
- **核心贡献：** 在深度网络训练中系统验证 **Momentum** 与 **Nesterov momentum** 对 RNN/LSTM 收敛速度与泛化的关键作用；给出 $\mu \approx 0.9$ 等实践默认值。
- **对 wiki 的映射：**
  - [SGD Momentum](../../wiki/methods/sgd-momentum.md)
  - [Nesterov Momentum](../../wiki/methods/nesterov-momentum.md)

### 5) Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (Duchi et al., 2011)
- **链接：** <https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>
- **核心贡献：** **Adagrad**：按参数维度累积梯度平方并做 per-parameter 学习率缩放，适合稀疏特征；但学习率单调衰减可能过早停滞。
- **关键更新：** $G_t = G_{t-1} + g_t \odot g_t$，$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon} \odot g_t$。
- **对 wiki 的映射：**
  - [Adagrad](../../wiki/methods/adagrad.md)

### 6) Lecture 6.5 — rmsprop: Divide the gradient by a running average of its recent magnitude (Tieleman & Hinton, 2012)
- **链接：** <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>（CSC321 课程讲义）
- **核心贡献：** **RMSProp**：用梯度平方的指数滑动平均归一化步长，缓解 Adagrad 学习率持续衰减问题；成为 Adam 的直接前驱之一。
- **对 wiki 的映射：**
  - [RMSProp](../../wiki/methods/rmsprop.md)

### 7) ADADELTA: An Adaptive Learning Rate Method (Zeiler, 2012)
- **链接：** <https://arxiv.org/abs/1212.5701>
- **核心贡献：** **Adadelta**：用梯度更新量的 RMS 与参数更新量的 RMS 之比自适应缩放步长，**无需手动全局学习率**。
- **对 wiki 的映射：**
  - [Adadelta](../../wiki/methods/adadelta.md)

### 8) Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)
- **链接：** <https://arxiv.org/abs/1412.6980>
- **核心贡献：** **Adam**：结合 Momentum 一阶矩与 RMSProp 二阶矩估计，偏差校正后做 per-parameter 自适应步长；深度学习默认优化器之一。
- **关键更新：** $m_t, v_t$ 为 $g_t, g_t^2$ 的 EMA；$\hat{m}_t, \hat{v}_t$ 偏差校正；$\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t}+\epsilon)$。
- **对 wiki 的映射：**
  - [Adam](../../wiki/methods/adam.md)

### 9) Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
- **链接：** <https://arxiv.org/abs/1711.05101>
- **核心贡献：** **AdamW**：将 **权重衰减（weight decay）** 从 Adam 的梯度自适应项中解耦，按 $\theta \leftarrow \theta - \eta\lambda\theta$ 独立施加；修正 Adam + L2 在自适应优化器下的正则化语义，成为 Transformer 预训练事实标准。
- **对 wiki 的映射：**
  - [AdamW](../../wiki/methods/adamw.md)

### 10) Symbolic Discovery of Optimization Algorithms (Chen et al., 2023)
- **链接：** <https://arxiv.org/abs/2302.06675>
- **核心贡献：** 用符号搜索发现 **Lion** 优化器：仅维护一阶动量，用 **符号函数 sign** 更新参数，内存与计算更省；在部分视觉与语言任务上报告与 AdamW 可比或更优的样本效率。
- **对 wiki 的映射：**
  - [Lion](../../wiki/methods/lion.md)

### 11) Stochastic Gradient Descent Tricks (Bottou, 2010)
- **链接：** <https://leon.bottou.org/publications/pdf/tricks-2010.pdf>
- **核心贡献：** 系统总结 **mini-batch SGD** 在机器学习中的实践技巧：学习率调度、动量、平均化等；连接经典 SGD 与现代深度学习训练。
- **对 wiki 的映射：**
  - [SGD](../../wiki/methods/sgd.md)
  - [Deep Learning Optimizers 对比](../../wiki/comparisons/deep-learning-optimizers.md)

## 辅助一手资料

- [Understanding Deep Learning (Prince, 2023)](../../sources/books/udl_book.md) — 第 6 章 *Training Models* 统一讲解 SGD、Momentum、Adam 等。
- [PyTorch 官方站点与文档索引](../../sources/repos/pytorch-official.md) — `torch.optim` 各优化器 API 与默认超参。

## 当前提炼状态

- [x] 9 个优化器各有一条以上一手出处
- [x] 映射到独立 wiki method 页与对比页
- [ ] 后续补：各优化器在机器人 RL / VLA 预训练中的实测对比数据
