# AutoVLA-D: Learn a Trajectory-Geometric Condition from Reasoning for Autonomous Driving VLA（arXiv:2611.12382）

> 一手论文原文归档；关键术语与结果均以 arXiv v1 原文为依据，不把作者报告视为本地复现实验。

- **类型：** paper / autonomous-driving / vision-language-action / trajectory-planning
- **英文标题：** AutoVLA-D: Learn a Trajectory-Geometric Condition from Reasoning for Autonomous Driving VLA
- **中文标题：** AutoVLA-D：从驾驶推理学习轨迹几何条件的自动驾驶 VLA
- **作者：** Yuguang Yang, Zhewen Tan, Canyu Chen, Linlin Yang, Cheng Chi, Chunyang Liu, Kehua Sheng, Jinyu Yang, Baochang Zhang, Bo Zhang, Yan Wang, Xianbin Cao
- **机构：** 北京航空航天大学、清华大学智能产业研究院、滴滴、中国传媒大学、中国人民大学、哈尔滨工业大学（深圳）
- **会议：** NeurIPS 2026（论文及 arXiv 均标注已接收）
- **arXiv：** <https://arxiv.org/abs/2611.12382>
- **PDF：** <https://arxiv.org/pdf/2611.12382>
- **版本：** arXiv:2611.12382v1，2026-11-09
- **代码：** <https://github.com/ShivaTeam/AutoVLA-D>
- **官方仓库归档：** [AutoVLA-D 源码归档](../repos/autovla-d.md)
- **wiki 映射：** [AutoVLA-D 独立论文实体](../../wiki/entities/paper-autovla-d.md)
- **入库与核对日期：** 2026-11-19
- **核对范围：** arXiv PDF v1：摘要、§1–5（任务形式化、1D-Brohan、两阶段 SFT、GRPO、TrajCond 分析、语义消融、结论与局限）；Figure 1–4 / Table 1–6；Supplement A–H 中的基准协议、提示模板、轨迹离散化算法及附加实验 Table 7–14 / Figure 6–9；并交叉核对当前 GitHub README、数据处理、词表扩展、codebook 构建与训练配置。
- **通读：** 已完成
- **一句话说明：** 将从驾驶推理学习的离散动作 token 轨迹作为 TrajCond 中间条件，最终仍预测连续文本航点，改善驾驶 VLA 指令语义与轨迹几何间的对齐。

## 论文精读要点

### 问题与任务（§1–3.1）
- 直接把“减速、让行、转弯”等动作性语言映射为绝对连续自车状态，缺少易学习的局部运动接口。
- 连续轨迹记为 `τ={(x_t,y_t,θ_t)}`，最终监督目标是按时间步标准化的数值航点；离散 `a` **不是最终控制轨迹**。
- TrajCond 表示逐步自车差分运动 `(Δx, Δy, Δyaw)`，平移分量转换到上一步自车朝向的局部坐标系。

### 离散 token 与阶段训练（§3.2–3.3）
- 对比共用 3D codebook 的 K-Means、K-Disks 与 x/y/yaw 分轴的 **1D-Brohan**；主设置共 **768** tokens，各轴 **256**。
- 论文算法描述为按训练数据 **1%–99% 分位区间**等间隔分桶，每一步每轴选近邻 4 个并枚举 64 个组合，以车辆角点误差选最佳动作 token 组；**当前开源 `build_codebook_from_gt.py` 的分桶边界实际用全量 min/max**，重现实验要先核清这一不一致。
- Stage 1 学习从图像、指令、历史运动和可用 CoT 预测动作 token；Stage 2 从 Stage-1 权重初始化，先生成 TrajCond，再输出 CoT 与连续轨迹（NAVSIM 顺序 `AT → CoT → PT`）。
- NAVSIM 使用 Qwen3-VL-235B 生成的 CoT 标注，含 critical objects、explanation、meta behaviour；nuScenes / NeuroNCAP 无 CoT、无 GRPO。NAVSIM 另用 GRPO 作为可选 RFT，不能把它写成所有数据集均采用。

### 核心数据（§4、Tables 1–6、Supplement Tables 9–14）
| 场景与协议 | 原文作者报告 | 读法 |
|---|---|---|
| NAVSIM navtest，1 个摄像头，PDMS | **92.0** | Qwen3-VL-4B，包含 GRPO，Table 1 |
| NAVSIM navtest，EPDMS | **86.8** | NAVSIM v2 修订版指标，Table 2；不能与 PDMS 混用 |
| NAVSIM 纯连续 6 ep. → 两阶段 Brohan 3+3 ep. | **88.5 → 90.8 PDMS** | 两者都为 SFT-only，Supplement Table 9 |
| NAVSIM 纯连续 + GRPO → 两阶段 Brohan + GRPO | **89.7 → 92.0 PDMS** | 对齐后训练协议，Supplement Table 9 |
| nuScenes ST-P3，Qwen3-VL-4B | **L2 0.30 m、碰撞率 0.09%** | 开环，Table 3；不含 CoT/GRPO |
| NeuroNCAP，Qwen2.5-VL-3B | **score 3.63、collision 0.36** | 仿真闭环，Table 4；不可和 NAVSIM 分数直接比较 |
| 768-token 1D-Brohan 轨迹 4 秒重建误差 | **test 1.56 cm** | Supplement Table 10；对应 tokenization，不代表在线预测误差 |

### 分析与局限
- 离散 token **直接作为最终输出**通常弱于连续航点；作为中间 TrajCond 则更有用，效果不是单纯“输出更长”或“增加辅助损失”。
- Table 6：用 pause token 替换 Stage-2 的显式推理，SFT 可能仍保持较好条件预测，但该设置进行 GRPO 会退化；因此 **TrajCond 不能在 RFT 中完全代替显式推理**。
- 作者明确列出额外训练/解码成本与少见 token 的学习不足；未给出车端实时性或真实道路安全认证结果。
- 论文方法按分位数、当前公开 codebook 脚本按 min/max：实现和论文的对应关系需复核，不能直接声称完全一致。

## 对 wiki 的映射

- [AutoVLA-D 独立详情](../../wiki/entities/paper-autovla-d.md)：方法、流程图、实验对照与复现风险。
- [DriveTeach-VLA](../../wiki/entities/paper-driveteach-vla.md)：同为 ShivaTeam 驾驶 VLA，但前者用空间视觉提示，AutoVLA-D 用动作 token 作为轨迹条件。
- [VLA 方法](../../wiki/methods/vla.md)：离散动作 tokenizer 与自回归连续轨迹双重表示的技术实例。
