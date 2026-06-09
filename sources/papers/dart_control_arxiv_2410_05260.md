# DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control（arXiv:2410.05260）

> 来源归档（ingest）

- **标题：** DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control（简称 **DART**）
- **类型：** paper / text-to-motion / diffusion / autoregressive / spatial-control / human-motion
- **arXiv abs：** <https://arxiv.org/abs/2410.05260>
- **PDF：** <https://arxiv.org/pdf/2410.05260>
- **项目页：** <https://zkf1997.github.io/DART/> — 归档见 [`sources/sites/dart-control-project.md`](../sites/dart-control-project.md)
- **代码：** <https://github.com/zkf1997/DART> — 归档见 [`sources/repos/zkf1997_dart.md`](../repos/zkf1997_dart.md)
- **作者：** Kaifeng Zhao, Gen Li, Siyu Tang（ETH Zürich）
- **会议：** ICLR 2025
- **入库日期：** 2026-06-09
- **一句话说明：** **自回归运动原语 + 潜扩散**：在 **在线文本流** 与 **运动历史** 条件下实时合成任意长人体运动，并在同一潜空间用 **噪声优化** 或 **RL 策略** 实现 **空间目标/场景几何** 约束控制。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 训练数据 | [AMASS](https://amass.is.tue.mpg.de/) | 大规模 SMPL 人体动捕；VAE 压缩可缓解原始数据抖动/毛刺 |
| 离线长序列组合 | FlowMDM [Barquero et al., 2024] | 需预知全时间线、离线慢速；DART 宣称 **~10×** 生成速度且在线 |
| 潜空间编辑 | DNO [Karunratanakul et al., 2024b] | 全序列显式表示上的扩散噪声控制；DART 原语空间在 **语义对齐** 上更优 |
| 并发自回归扩散 | CloSD [Tevet et al., 2025] | 目标关节位置条件 + 配对控制数据；DART 仅用 **motion-only** 数据学潜空间再后验控制 |
| 物理后处理 | PHC（Perpetual Humanoid Control） | 项目页演示：DART 运动学输出 + PHC 物理跟踪可减轻穿模/滑步 |

## 摘要级要点

- **问题：** 主流 text-to-motion 生成 **短、孤立** 片段；**长时连续**、**在线文本流**、**空间约束**（关键点/路点/场景 SDF）三者同时满足仍难。
- **表示：** 重叠 **运动原语** $\mathbf{P}^i=[\mathbf{H}^i,\mathbf{X}^i]$：历史 $H{=}2$ 帧 + 未来 $F{=}8$ 帧；SMPL-X 过参数化 **276 维/帧**（根位姿、局部旋转、关节位置、一阶差分）；原语在骨盆局部坐标系 canonicalize。
- **生成骨干：** **运动原语 VAE**（Transformer，借鉴 MLD）压缩未来帧 → **潜扩散去噪器**（CLIP 文本 + 历史条件，**10 步** DDPM/DDIM + classifier-free guidance）；**scheduled training** 渐进引入测试时历史分布以稳定长 rollout。
- **效率：** 单卡 RTX 4090 **>300 FPS** 自回归生成；RL 路点控制 **~240 FPS**。
- **空间控制：** 将 DDIM 采样视为 $\mathbf{Z}_T \mapsto$ 运动 的确定性映射，在潜噪声上最小化目标距离 + 场景/物理正则：
  - **梯度优化**（Alg. 2）：in-between、人体–场景交互（SDF 足地接触/碰撞）
  - **RL 策略**：潜噪声为动作空间，文本条件路点到达
- **语义注意：** 粗粒度 **整句多动作** 标签会导致原语级 **随机跳转**；需拆成 **逐动作短 prompt** 序列。

## 核心摘录（面向 wiki 编译）

### 1) 自回归 rollout（Alg. 1 归纳）

给定种子历史 $\mathbf{H}_{seed}$ 与在线文本序列 $C=[c^1,\ldots,c^N]$：每步对当前历史 + $c^i$ 采样潜变量 $\hat{\mathbf{z}}_0^i$，解码未来 $\hat{\mathbf{X}}^i$，拼接序列，并用末 $H$ 帧更新下一原语历史。

### 2) 潜空间控制（Eq. 2–3）

原始运动空间优化易产出非自然样本；DART 在 **已学 realistic 潜空间** 上优化初始噪声列表 $\mathbf{Z}_T$，目标为 $\mathcal{F}(\Pi(\text{rollout}(\mathbf{Z}_T)), g) + cons(\text{rollout}(\cdot))$。

### 3) 与机器人知识库的关系

- **上游人体先验：** SMPL-X 轨迹可经 GMR 等接人形/动画管线；运动学 artifact 可用 PHC 类物理跟踪精炼。
- **与 HY-Motion / GENMO 对照：** 同属 **扩散式人体运动**；DART 强调 **原语自回归 + 在线文本 + 潜空间空间控制**，而非单次 clip 生成或估计–生成统一。
- **勿与 WBC 文献中的 “DART” 混淆：** `wiki/comparisons/wbc-vs-rl.md` 中 “DART（MPC 教神经网络）” 指 **另一套蒸馏范式**，非本论文。

## 对 wiki 的映射

- 沉淀方法页：[dart-control](../../wiki/methods/dart-control.md)
- 交叉（见方法页「关联页面」）：`wiki/methods/diffusion-motion-generation.md`、`wiki/methods/hy-motion-1.md`、`wiki/methods/genmo.md`、`wiki/entities/awesome-text-to-motion-zilize.md`、`wiki/entities/phc.md`、`wiki/entities/amass.md`
