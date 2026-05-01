# universal_skeleton

> 来源归档（ingest）

- **标题：** Universal Skeleton-Based Action Recognition: Heterogeneous Open-Vocabulary Learning via Multi-Grained Motion-Text Alignment
- **类型：** paper
- **作者：** Jidong Kuang, Hongsong Wang, Jie Gui（东南大学网络空间安全学院）
- **arXiv：** 2604.17013
- **GitHub：** https://github.com/jidongkuang/Universal-Skeleton
- **入库日期：** 2026-05-01
- **沉淀到 wiki：** 是 → [`wiki/methods/skeleton-action-recognition.md`](../../wiki/methods/skeleton-action-recognition.md)

## 一句话说明

提出 HOVL（Heterogeneous Open-Vocabulary Learning）框架，通过**多粒度动作-文本对齐**在异构骨架数据上实现跨数据集、开放词汇的动作识别，为人形机器人跨形态动作理解提供统一表示。

## 为什么值得保留

1. **直接对应人形机器人多形态问题**：不同人形机器人骨骼结构各异（关节数、连接方式不同），HOVL 的异构统一表示思路可迁移到跨机器人迁移学习场景。
2. **开放词汇 = 零样本泛化**：模型无需对每个新动作类别重新训练，和 VLA / CLAW 语言对齐思路高度呼应。
3. **引入 HOV 大规模基准数据集**：整合 NTU RGB+D 120 与 HumanML3D，覆盖 3D 骨架与 2D 姿态序列两种格式，长尾多标签评测设置。
4. **多粒度对比学习**：三层对齐（全局/流级/帧级）对理解动作-语言对应关系有方法论参考价值。

## 核心摘录

### 问题背景

现有骨架动作识别方法存在两大局限：
- **封闭词汇**：只能识别训练集内的动作类别，无法处理新类别
- **同构假设**：假设所有骨架拓扑相同，无法跨数据集 / 跨机器人迁移

HOVL 同时攻克这两个限制。

### 方法：HOVL（Heterogeneous Open-Vocabulary Learning）

**架构三组件：**

1. **统一骨架表示模块（Unified Skeleton Representation）**
   - 将不同拓扑结构（NTU 25 关节、HumanML3D 22 关节等）映射到统一空间
   - 消除骨架结构异构性带来的特征空间不一致

2. **多流时空动作编码器（Multi-stream Spatio-temporal Motion Encoder）**
   - 基于 Transformer，提取不同骨架表示流（3D 关节位置、2D 姿态等）的时空特征
   - 输出多模态骨架嵌入

3. **多粒度动作-文本对齐（Multi-Grained Motion-Text Alignment）**
   基于 CLIP 对比学习框架，三层对齐：
   - **全局实例对齐**：整段动作序列 ↔ 文本描述（动作类别名 / 自然语言）
   - **流级对齐**：各骨架流特征 ↔ 文本模态
   - **细粒度对齐**：帧级 / token 级精细对应

### 数据集：HOV Skeleton Dataset

| 来源数据集 | 骨架格式 | 规模 |
|-----------|---------|------|
| NTU RGB+D 120 | 3D 关节序列 | 120 类、57,600 样本 |
| NTU RGB+D 120 | 2D 姿态序列 | 同上（不同格式） |
| HumanML3D | 3D 动作序列 | 14,616 动作、44,970 标注 |

评测设置：长尾多标签（Long-tail Multi-label），贴近真实世界分布。

### 代码结构

```
Universal-Skeleton/
├── datasets/         # HOV 数据集加载与预处理
├── models/           # HOVL 模型架构
│   └── ...
├── third_party/
│   └── clip/         # CLIP 视觉-语言预训练模块
├── scripts/          # 训练 / 评测脚本
└── utils/            # 工具函数
```

## 与本知识库的对应关系

本 source 文件的详细 wiki 映射已在沉淀页 `wiki/methods/skeleton-action-recognition.md` 中给出全部交叉引用。

主要对应关系：
- 开放词汇动作识别 → `wiki/methods/skeleton-action-recognition.md`（主沉淀页）
- 动作-语言对齐 → `wiki/methods/claw.md`、`wiki/methods/vla.md`
- 跨机器人骨架迁移 → `wiki/methods/motion-retargeting-gmr.md`
- 模仿学习 / 演示数据 → `wiki/methods/imitation-learning.md`

## 当前提炼状态

- [x] 论文背景与问题定义
- [x] HOVL 架构三组件
- [x] HOV 数据集构成
- [x] 与本项目已有 wiki 页面的对应关系
- [ ] 后续可深挖：PoE 风格骨架表示统一、异构 Transformer 形式化、Zero-shot 动作泛化到机器人形态
