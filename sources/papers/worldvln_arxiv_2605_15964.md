# WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation（arXiv:2605.15964）

> 来源归档（ingest）

- **标题：** WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation
- **类型：** paper / aerial VLN / world action model / UAV
- **arXiv：** <https://arxiv.org/abs/2605.15964>
- **PDF：** <https://arxiv.org/pdf/2605.15964>
- **项目页：** <https://embodiedcity.github.io/WorldVLN/>
- **代码：** <https://github.com/EmbodiedCity/WorldVLN.code>（项目页官方链接；同名组织下亦有 <https://github.com/EmbodiedCity/WorldVLN>）
- **机构：** 清华大学、山东大学、Manifold AI、北京理工大学、东北大学等（EmbodiedCity 相关）
- **入库日期：** 2026-05-24
- **一句话说明：** 将 **空中 VLN** 表述为 **预测驱动的 world–action 问题**：在 **潜空间自回归视频骨干** 上预测 **短视界世界状态转移**，经 **动作解码器** 直接输出 **可执行航点**；每段动作执行后把新观测编码回上下文形成闭环；**两阶段训练**（SFT 接地 + **Action-aware GRPO**）在室内外 UAV 基准上相对 VLA 基线 **成功率提升 12%+**，并报告 **真机无人机零样本迁移**。

## 摘要级要点（与 arXiv abstract 一致）

- **问题重框：** 空中 VLN 需要 **闭环感知–动作**；作者认为应 **预期自身动作下的世界演化**，再据此选动作，而非仅做「指令 + 观测 → 动作」的条件映射。
- **与整段视频生成的张力：** 双向生成整段 clip 与 **observe–act–update** 因果闭环不匹配；空中 VLN 的大视角变化与累积误差更需要 **持久记忆与在线修正**。
- **WorldVLN 结构：** 复用 **预训练视频潜自回归 Transformer**；预测 **短视界潜世界转移** → **waypoint 动作** → 执行后 **新观测写回** 自回归上下文。
- **训练：** Stage 1 监督：指令–视频对接地导航动力学 + 视频–轨迹对训练动作解码器；Stage 2 **Action-aware GRPO**（作者称首个面向 **自回归 WAM** 的 RL）：在线自回归 rollout，**段级奖励**（轨迹精度、任务进展、参考策略正则）+ **时间衰减** 强调早期决策对下游的影响。
- **实验：** 公开 **室外 + 室内** UAV 基准（项目页点名 **UAV-Flow**、**IndoorUAV**）；相对现有 **VLA** 基线一致更优，困难样本优势更大；**真实无人机部署**（室内外视频于项目页）。

## 核心摘录（面向 wiki 编译）

### 与 VLA / imagine-and-rank 的对比（论文 Related Work 归纳）

| 路线 | 特点 | WorldVLN 立场 |
|------|------|----------------|
| **VLA** | VLM 语义强，但少显式 **动作条件世界动力学** | 缺时空–因果结构，易把导航当条件映射 |
| **整段视频世界模型 + 视觉里程计** | 先生成未来帧再反推动作 | 结构/目标与闭环导航错位，代价高 |
| **imagine-and-rank** | 多候选路线 rollout 再排序 | 间接、算力贵 |
| **自回归 WAM** | 潜世界预测与动作 **同策略内耦合** | WorldVLN 针对 **空中 VLN** 首提该形态 + **Action-aware GRPO** |

### 两阶段训练（项目页 Training 区块）

1. **Stage 1（SFT）：** 潜自回归骨干 + 指令–视频；动作解码器 + 视频–轨迹。
2. **Stage 2（Action-aware GRPO）：** 多 rollout 采样 → 段级奖励 → 时间衰减 credit assignment → 超越纯 SFT 的动作执行（项目页 Training Analysis 定性结论）。

### 评测语境（索引级）

- **室外仿真：** UAV-Flow（项目页 Demo 标题）
- **室内仿真：** IndoorUAV
- **真机：** Outdoor / Indoor Real-World Deployment（项目页视频区）

## 对 wiki 的映射

- 沉淀实体页：[WorldVLN（空中 VLN · 自回归 WAM）](../../wiki/entities/paper-worldvln-aerial-vln-wam.md)
- 交叉补强：[视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)、[World Action Models（WAM）](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[机器人世界模型训练闭环 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)

## 全文通读复核（2026-11-13）

- **核对范围：** arXiv:2605.15964v1 正文 §1–§6、附录 A.1–A.9（架构、训练、基准、局限、真机）；图表按文字层核对，参考文献浏览；项目页、官方 README、在线服务入口/会话与动作解码相关源码；未运行模型或观看视频。
- **通读：** 已完成
- InfinityStar-8B + Wan VAE 多尺度离散 latent：49 raw frames=1+3×16，每段 16 帧对应四个 latent 时间位置。SFT 用真实历史 teacher forcing；部署每段动作执行后以实测视觉更新，不把预测未来一直滚成真实历史。
- 动作 teacher 先由视频–轨迹训练，冻结视觉编码后蒸馏 latent adapter，再训练完整 action decoder；部署不要求先渲染完整 RGB 视频再做 VO，但公开调试服务默认可保存预测视频。
- GRPO G=4，轨迹/任务/参考权重 .2/.7/.1，γ=.9，KL=.9、clip=.02，原文冻结前五 model chunks；当前仓另有全骨干训练入口，不能反写成论文实验配置。未复现 RL。
- UAV-Flow Fixed/Open 平均 SR 79.12/78.02%，相对最强基线 +13.51/+12.24 个百分点；IndoorUAV 41.76 vs π0 27.16（+14.60pp），不写相对增长 12%。类别并非全胜，如 Pass/Rotate 等仍低于部分基线。
- 真机为 250mm 四旋翼、C270、Orin NX16GB 通信、PX4 低层；高层在地面站服务器，室内14相机MoCap、室外GPS/TFmini-S用于低层，不输入WAM。正文只给部署案例，无完整量化真机成功率表；无广泛多种子统计。
- 代码当前非占位，见 repo 复核；API 是 6D cm/deg 增量，论文任务抽象为4D平移+yaw，必须单独核对坐标/单位契约。
