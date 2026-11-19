# Context and Diversity Matter: The Emergence of In-Context Learning in World Models（arXiv:2509.22353 / ICLR 2026）

> 来源归档（论文全文精读 + 索引溯源）

- **英文标题：** Context and Diversity Matter: The Emergence of In-Context Learning in World Models
- **列表来源：** [Awesome World Models](https://github.com/sun254667/awesome-world-models)（Theory Interpretability & Positions，#555/571）
- **发表：** ICLR 2026（arXiv:2509.22353v2 [cs.LG] 27 Feb 2026）
- **作者：** Fan Wang*,†, Zhiyuan Chen*, Yuxuan Zhong*, Sunjian Zheng, Pengtao Shao, Bo Yu, Shaoshan Liu†, Jianan Wang, Ning Ding, Yang Cao, Yu Kang（AIRS-CUHK / 中科大 USTC / Astribot）
- **arXiv：** <https://arxiv.org/abs/2509.22353>
- **代码：** <https://github.com/airs-cuhk/airsoul/tree/main/projects/MazeWorld>（[`../repos/airs-cuhk-airsoul.md`](../repos/airs-cuhk-airsoul.md)）
- **入库日期：** 2026-08-10
- **再核日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2509.22353v2` 原文全篇（2419 行，含 §1–§6 正文、Theorem 1 证明思路、Appendix 算法/实验/消融）及官方仓库 `airsoul/projects/MazeWorld/`
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md`](../../wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md)

---

## 核心论文精要

### 1. 世界模型 ICL 的形式化与双机制分离

- 将世界模型从单步 `T_2` 扩展到更长上下文 `T_1 > T_2` 下的预测误差下降严格定义为 World-Model ICL。
- 识别出两种截然不同的泛化机制：
  1. **环境识别（Environment Recognition, ER）：** 依赖权重内记忆有限训练环境集合 `E`（IWL），上下文只负责从 `E` 中识别当前匹配度最高的环境后验 `p(e | C_T)`；受限于最优匹配误差（BME），即使 `T → ∞`，新环境的渐近误差也不会消失。
  2. **环境学习（Environment Learning, EL）：** 直接将环境动力学编码进长上下文工作记忆，等价于在未知 MDP 上做非参数贝叶斯密度积累 `p(q_t, o_{t+1} | C_T) / p(q_t | C_T)`。

### 2. Theorem 1 渐近误差上界

- **ER 上界：** `TV(p̂_ER, p_{e0}) ≤ min[ (α/3)(|E|-1) T^{-1/2}, max TV ] + min_{e ∈ E} TV(p̂_{θ,e}, p_{e0})` —— 当训练环境数量 `|E|` 较小时快速匹配收敛，但存在不可消除的 `BME` 下限；若环境多而长上下文缺失或能力受限，无法靠识别覆盖新分布。
- **EL 上界：** 当上下文 `T > 4|S|^2|A|^2 log(4|S||A|/δ)` 时，`TV(p̂_EL, p_{e0}) ≤ sqrt(2|O||S||A| log(4|O|/δ)) · T^{-1/2}`，误差随上下文增长稳定趋零（理论转折由高环境多样性 `|E|` 与足够长上下文 `T` 共同触发）。

### 3. L2World 架构与三套基准实验

- **L2World（Learn-to-WorldModel）：** `ResNet VAE` 视觉编码/解码器（Cart-Pole 为 MLP）+ 18 层 **Gated Slot Attention（GSA）** 序列主干（训练期按 chunk 并行、推理期递归维护槽记忆）+ 高斯潜状态转移解码头 `N(ŝ_t, σ̂_{s,t})`。
- **Cart-Pole 动力学实验：** 对比 `|E| ∈ {1, 4, 16, 8K}`（128K 轨迹 × 200 步）；低多样性（1/4-Env）表现为 ER 且在约 `T=10` 后饱和于 `error=BME` 线上；`8K-Envs` 激活 EL，误差越过 BME 线持续下降，且过拟合小数据集会破坏 OOD EL 泛化。
- **Procedural Mazes（320M 帧）与 ProcTHOR：** `Maze-32K-L`（3.2 万个 15×15 迷宫 × 10K 步）训练下，未见迷宫 1 步预测 PSNR 从 `T=1` 的 `16.37` 随上下文升至 `T=10K` 的 `24.65`（对比 `Maze-128-L` `21.52`、`Maze-32K-S` `20.31`、Dreamer `22.12`、NWM `17.85`）；先预训练 `Maze-32K-L` 再用 `ProcTHOR-40K`（2K 步）微调在未见房屋 `T=1000` 达 `25.40` PSNR（Dreamer `21.02`、NWM `20.08`）。
- **可解释性探针：** 上下文乱序对 EL（`32K-L`）破坏显著大于 ER（`128-L`）；GSA 第 12/18 层 Slot 记忆 t-SNE Silhouette 分数（`0.53/0.54`）与预测编码替代实验显示模型通过长程历史在内部形成空间结构与目标编码。

## 开源与仓库边界（步骤 2.5）

| 项 | 状态 | 备注 |
|----|------|------|
| 论文原文 | 已开源（arXiv v2） | 含全部定理证明、消融及架构细节 |
| 官方实现（`airsoul/projects/MazeWorld`） | **已开源（GitHub）** | 对应 `E2EObjNavSA`（VAE + `POTARDecisionModel` / GSA）训练、验证、生成与交互评测脚本 |
| 依赖环境 | 外部依赖 | 需要 `FutureAGI/Xenoverse` / `l3c.mazeworld` 迷宫环境与数据 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md`](../../wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md)
- 代码仓：[`../repos/airs-cuhk-airsoul.md`](../repos/airs-cuhk-airsoul.md)
- 列表实体：[`wiki/entities/awesome-world-models.md`](../../wiki/entities/awesome-world-models.md)
