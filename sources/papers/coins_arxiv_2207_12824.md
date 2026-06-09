# COINS: Compositional Human-Scene Interaction Synthesis with Semantic Control（arXiv:2207.12824）

> 来源归档（ingest）

- **标题：** Compositional Human-Scene Interaction Synthesis with Semantic Control（COINS）
- **类型：** paper / human-scene-interaction / generative-model / synthetic-data / computer-graphics
- **arXiv abs：** <https://arxiv.org/abs/2207.12824>
- **会议：** ECCV 2022
- **项目页：** <https://zkf1997.github.io/COINS/index.html> — 归档见 [`sources/sites/coins-zkf1997-github-io.md`](../sites/coins-zkf1997-github-io.md)
- **代码：** <https://github.com/zkf1997/COINS> — 归档见 [`sources/repos/coins.md`](../repos/coins.md)
- **机构：** ETH Zürich；Google（Thabo Beeler）
- **入库日期：** 2026-06-09
- **一句话说明：** 给定 3D 场景与 **动作–物体实例** 语义规格（如「sit on the chair」），用 **Transformer cVAE** 分阶段生成 **SMPL-X 人体** 与场景的自然交互；仅用 **原子交互** 训练即可 **组合** 出未见过的复合交互（如「sit on sofa + touch table」），并发布 **PROX-S** 数据集扩展。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://zkf1997.github.io/COINS/index.html> | 交互 demo、PROX-S、定性对比与视频 |
| 代码仓库 | <https://github.com/zkf1997/COINS> | PelvisVAE/BodyVAE 训练与采样、PROX-S 加载、PiGraph-X/POSA-I 基线 |
| 基础数据集 | [PROX](https://prox.is.tue.mpg.de/index.html) | 人–场景交互 MoCap + 3D 场景约束 |
| 场景扩展 | [PROX-E / PSI](https://github.com/yz-cnsdqz/PSI-release) | 场景语义与实例分割来源之一 |
| 对比基线 | [POSA](https://posa.is.tue.mpg.de/index.html) | 人–场景放置与接触特征；COINS 实现 POSA-I 变体 |
| 下游关联 | [CRISP](../../wiki/methods/crisp-real2sim.md) | 同在 PROX 等人-centric 基准生态；COINS 走 **生成式场景填充**，CRISP 走 **单目 Real2Sim** |

## 摘要级要点

- **问题：** 现有方法多建模 **几何邻近**（人体与场景碰撞/接触），忽略 **高层交互语义**（具体动作 + 具体物体实例），难以按「坐在哪把椅子上」等规格控制虚拟人行为。
- **语义规格：** 交互语义表示为 **可变数量的原子 (action, object_instance) 对**，如 `sit on-chair`、`touch-table`；复合交互为多个原子组合。
- **表示：** 人体用降采样 **SMPL-X mesh**（655 顶点 + 二值接触特征）；场景为 **物体实例点云**（8192 点 × 位置/颜色/法向）；动作用 **可学习 embedding** 以 positional encoding 注入对应物体 token。
- **三阶段生成：** (1) **PelvisVAE** 推断骨盆位置与朝向；(2) **BodyVAE** 在骨盆局部坐标系生成 mesh + 接触图；(3) **交互优化** 用接触特征与场景 SDF 精修 SMPL-X 参数，减少穿透。
- **组合生成（Ours-C）：** 仅用原子交互数据训练，推理时 (a) 多原子骨盆分布 **求交/平均**；(b) BodyVAE 用 **接触统计导出的 attention mask** 在部位级组合原子交互——**无需复合交互训练数据**。
- **数据：** **PROX-S** = PROX + PROX-E 扩展，含 **场景实例分割**、**逐帧 SMPL-X 拟合**、**逐帧 action–object 语义标注**。
- **评测：** 感知研究 + 语义准确度/接触/无碰撞/多样性指标；相对 **PiGraph-X**、**POSA-I** 显著更优；组合任务上优于 **PiGraph-X-C**。

## 核心摘录（面向 wiki 编译）

### 1) 组合交互表示

- 交互 $\mathbf{I}=(\mathbf{B},\{(a^{i},o^{i})\}_{i=1}^{M})$：人体 $\mathbf{B}$ + $M$ 个 **原子** (动作 one-hot, 物体实例点云)。
- **原子交互** 不可再分（如「sit on a chair」）；**复合交互** 含多个原子（如「sit on chair and type on keyboard」）。
- 物体用 **实例级** 点云而非仅类别标签，消解同场景多把椅子等 **实例歧义**。

### 2) Transformer cVAE 架构（PelvisVAE / BodyVAE）

| 模块 | 输入 | 输出 | 要点 |
|------|------|------|------|
| **PelvisVAE** | 场景坐标系下 action–object 对 | 骨盆位置 + 朝向分布 | PointNet++ 256 keypoints；动作 embedding 加到配对物体 token |
| **BodyVAE** | 骨盆局部系下 action–object 对 + 模板人体 | mesh 顶点 + 接触特征 | 共享 transformer 架构；latent 经 encoder-decoder attention 注入 |

- 动作 **不单独成 token**，而是 **positional encoding 加到对应物体 token**，使「sit on」只影响沙发而非桌子。
- BodyVAE 并联 **MLP regressor** 将采样 mesh 回归为 **SMPL-X 参数**，保证形体合法。

### 3) 交互优化（后处理）

- 优化 SMPL-X 的 $t, R, \theta$：$\mathcal{L}_{interaction}$（正接触顶点贴合物体）+ $\mathcal{L}_{coll}$（场景 SDF 碰撞）+ $\mathcal{L}_{reg}$（贴近初始化）。
- 与 PSI / PLACE / POSA 等同族 **物理合理性** 后处理。

### 4) 组合推理（无需复合训练数据）

| 步骤 | 机制 |
|------|------|
| **组合骨盆** | 各原子 PelvisVAE 解码分布采样 → 优化使骨盆帧空间接近 → 平均得复合骨盆 |
| **组合身体** | 由原子交互 **接触统计** 构造 transformer **attention mask**，在部位级组合 |

### 5) PROX-S 数据集

- **场景：** per-vertex **实例 ID** + **mpcat40** 类别（实例分割）。
- **交互：** 逐帧 **SMPL-X** + **action–object 对列表** 语义标注。
- 发布：Google Drive（见仓库 README）；需配合 PROX 原始 `scenes`/`sdf`/`body_segments` 与 PROX-E `scenes_semantics`。

### 6) 基线与主要结论（论文 Table 1–3 摘要）

| 对比 | COINS 优势 |
|------|------------|
| **PiGraph-X** | 关节独立、接触差、易穿透 |
| **POSA-I** | 放置优化易陷局部极小，语义不准 |
| **PiGraph-X-C**（组合） | Ours-C 语义准确度与无碰撞显著更好；样本效率远高于 25K 采样基线 |

## BibTeX（项目页提供）

```bibtex
@inproceedings{Zhao:ECCV:2022,
   title = {{COINS}: Compositional Human-Scene Interaction Synthesis with Semantic Control},
   author = {Zhao, Kaifeng and Wang, Shaofei and Zhang, Yan and Beeler, Thabo and Tang, Siyu},
   booktitle = {European conference on computer vision (ECCV)},
   year = {2022}
}
```

## 对 wiki 的映射

- 实体页：[COINS（论文）](../../wiki/entities/paper-coins-compositional-human-scene-interaction.md)
- 交叉：[CRISP](../../wiki/methods/crisp-real2sim.md)（PROX 生态）、[TokenHSI](../../wiki/entities/paper-bfm-38-tokenhsi.md)（人–场景交互 task token 化对照）
