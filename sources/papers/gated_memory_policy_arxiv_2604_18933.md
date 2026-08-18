# Gated Memory Policy: In-Context Memorization and Adaptation

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** Gated Memory Policy: In-Context Memorization and Adaptation
- **类型：** paper
- **状态：** arXiv 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2604.18933>
  - 项目页：https://gated-memory-policy.github.io/
- **代码：** https://gated-memory-policy.github.io/
- **作者：** Yihuai Gao, Jeff Jinyun Liu, Shuang Li, Shuran Song
- **机构：** Stanford University
- **入库日期：** 2026-08-18
- **一句话说明：** GMP（arXiv:2604.18933）：学习型内存门控+轻量 cross-attention 记忆；历史动作扩散噪声；MemMimic 非马尔可夫任务 SR +30.1%；代码/数据/部署说明已开。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 学「何时记、记什么」：门控决定是否激活历史上下文，cross-attention 构造潜记忆，并对历史动作加扩散噪声以抗噪。
- **对 wiki 的映射：**
  - [wiki/entities/paper-gated-memory-policy.md](../../wiki/entities/paper-gated-memory-policy.md)

### 方法与结果（归纳）

- **方法：** 二进制内存门控 + cross-attention 记忆模块 + 历史动作扩散噪声；推理时可缓存历史 token。
- **评测：** MemMimic 非马尔可夫基准平均 SR +30.1% vs 长历史基线；RoboMimic 马尔可夫任务保持竞争力。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-gated-memory-policy.md`
