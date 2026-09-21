# Scaling Laws for Neural Language Models

> 来源归档（深读 · arXiv:2001.08361）

- **标题：** Scaling Laws for Neural Language Models
- **作者：** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child 等（OpenAI）
- **类型：** paper / scaling-laws / nlp
- **arXiv：** <https://arxiv.org/abs/2001.08361>
- **入库日期：** 2026-09-21
- **一句话说明：** 语言模型 cross-entropy 损失随 **模型规模、数据规模、训练算力** 呈 **幂律** 缩放（跨 7+ 数量级）；给出固定算力预算下的 **最优模型/数据分配** 与过拟合规律——Light-O1 Transfer Scaling Law 的方法论对照基线。

## 核心摘录

### 1) 三轴幂律

- **要点：** $L(N,D,C)$ 对参数量 N、数据集 token 数 D、计算量 C 均为幂律；**宽/深等架构细节在宽范围内影响次要**。
- **对 wiki 的映射：** [`wiki/entities/paper-scaling-laws-neural-language-models.md`](../../wiki/entities/paper-scaling-laws-neural-language-models.md)

### 2) 算力最优分配

- **要点：** 固定 compute 下，**更大模型 + 更少步数** 往往优于小模型长跑；可估算 optimal N:D 比。
- **对 wiki 的映射：** [`wiki/concepts/embodied-scaling-laws.md`](../../wiki/concepts/embodied-scaling-laws.md)

### 3) 与具身 / Light-O1 的对照读法

- **要点：** Light-O1 将 human action pretraining 的 D（multimodal token）与适配后 held-out loss/MPJPE 做 **Kaplan 式幂律拟合** $L(D)=L_0+\alpha D^{-\eta}$；Kaplan 原文是 **LM cross-entropy**，但 **拟合协议与「堆数据是否可预测」** 同族。
- **对 wiki 的映射：** [`wiki/entities/light-o1.md`](../../wiki/entities/light-o1.md)

## 开源边界

| 状态 | 说明 |
|------|------|
| **无统一官方代码仓** | 论文为 OpenAI 内部实验总结；复现依赖公开 LM 栈 |
| **引用价值** | 作为 scaling law **方法论原典** 归档 |

## 参考来源

- 论文：<https://arxiv.org/abs/2001.08361>
