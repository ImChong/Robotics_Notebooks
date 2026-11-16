# FLORA: An Open Model Suite for Scaling Robotics Data Engines — 论文原文核对

> 来源归档（paper / arXiv v1）：以原始论文正文和附录作为一手依据；原文没有公开训练代码，不把论文训练细节当作公开实现。

- **标题：** FLORA: An Open Model Suite for Scaling Robotics Data Engines
- **作者：** Sicong Leng, Yunze Man, Zhiqi Li, Jing Wang, Yuhe Zhang, Ryan Punamiya, Siddharth Patki, Kaiyuan Zheng, Zi-ang Cao, Zhiding Yu, Sifei Liu, Danfei Xu, Ben Levin, Jan Kautz, Yuke Zhu, Johan Bjorck, Linxi “Jim” Fan
- **机构：** NVIDIA GEAR
- **发布日期：** 2026-10-22（arXiv v1 提交；与早期 PDF 预览日期区分）
- **论文 ID：** arXiv:2610.31779v1 [cs.RO]
- **论文原文：** <https://arxiv.org/html/2610.31779v1>
- **PDF：** <https://arxiv.org/pdf/2610.31779v1>
- **官方项目页：** <https://research.nvidia.com/labs/gear/flora/>
- **官方推理代码：** <https://github.com/NVlabs/Flora>
- **模型：** <https://huggingface.co/collections/nvidia/flora>
- **核对范围：** 原论文正文 §1 Overview、§2 Capabilities and Results、§3 Qualitative Examples、§4.1–§4.3 Methods、§5 Conclusion；Appendix A（Tables 2–5）、B（数据/评价口径）、C（reward 评估）、D（成本/吞吐假设）、E（定性失败案例）、F（窗口分段与两模型训练）、G（过程/结果/rubric 提示与 Rewarder 联合训练）、H（相关工作）。论文图 1–6、12–14 的语义与所引用表格/公式对应核对。
- **通读：** 已完成
- **一句话说明：** 分段器 + 动作意图 Captioner + 多接口 Rewarder 将机器人/人类视频变成可训练的数据标签；生产规模与开源样例能力必须区分。

## 正文逐节记录

### §1–§2 问题与贡献

- 数据标签覆盖三个用途：**带时间戳描述**（预训练语义条件、数据筛选和均衡）、**过程奖励**（RL 后训练中定位失败/步骤质量）、**结果奖励**（部署监测和轨迹评估）。
- 模型三个：Flora-TempSeg 负责片段起止；Flora-Captioner 负责每段意图描述；Flora-Rewarder 用同一 VLM 通过不同 prompt 生成 outcome、dense 或 rubric labels。
- WGO-Bench 共 100 集：HomER 25、DROID 50、Galaxea 25；主结果的平均按子集等权，tIoU ≥0.75。FLORA 分段 F1 **50.3**，gold-window caption accuracy **64.7**，joint semantic F1 **34.5**；Astra 分别 **46.0 / 77.4 / 39.0**。因此不能称 FLORA 在所有 captioning 指标领先。
- DenseCaptionBench 300 集：人类 100 / 平行夹爪机器人 100 / 灵巧手机器人 100；时间匹配阈值 0.5，caption 质量包含动作、物体、手部协同、关系/细节。Human100 联合 F1 **37.4 vs Astra 29.7**；Robot200 **51.0 vs 57.9**。
- RoboRewardBench 2,831 集 Outcome MAE×100 **40.0**，Astra **55.4**；Robometer OOD 567 条轨迹 Kendall τa×100 **72.0**，Astra **82.3**。DenseRewardBench 内部 40 集、4,861 秒级标签，失败 recall **53.3 vs 0.0**；RubricBench pooled L1×100 **6.5 vs 3.6**。

### §3 与 Appendix E：正反定性案例

- 论文展示 FLORA 相比通用基线能更完整覆盖高密度动作、保留手部交接与接线等细粒度操作意图，也能区分部分完成和彻底失败。
- **反例同样重要：** 过分割粗粒度事件、把 reaching 写成 placement、混淆物体和动作相位；部分机器人子集优劣顺序在只考察共同有效样本时逆转。视觉输入管线与评测协议不同，不能单因果归因到模型权重。

### §4.1 与 Appendix F：时间戳指令

- 定义：输入视频 X 和时长 T，输出时间有序的 `(s_i,e_i,c_i)`；其中 c 是动作**意图**，不是成功标志。
- 16 秒/64 帧（4 FPS）的重叠窗内，中心 12 秒为发布核心、两侧各 2 秒 guard；长机器人窗使用 32/24/4 秒配置。预测局部起止、跨窗延续标志，再投影到全局时间，过滤 guard-only 片段，按重叠/延续标志合并且允许并发区间重叠。
- Captioner 分别查看当前事件、最近且不重叠的相邻事件（before/target/after）；多机位在源时间点拼图，生成命令式操作指令，固定段界不回写。DenseCaptionBench 测试条件下前后文限制为每侧 2 秒，和生产系统相邻完整事件的上下文范围**不同**。
- §F.3 TempSeg 的 GRPO 奖励：`r_seg = 0.45 F_0.5 + 0.15 F_0.75 + 0.25 P_0.5 + 0.15 B`。B 衡量匹配区间端点差与延续标志，输出非法则得分 0，GRPO 只调节 LoRA。
- §F.4 Captioner 的 GRPO 奖励：`r_cap = 2 M/(R+M+F)`，包含参考要点被覆盖程度与事实性错误惩罚；可用来源证据支持要点，不奖励不受证据支持的臆测描述。

### §4.2 / Appendix G：三类奖励

- Outcome：任务提示 + 全集视频，输出 1–5 分，可选 grounded rationale；为排序读取 score-token 期望并归一化 `(E[s]-1)/4`，**不是任务成功概率**。最多取 64 帧，覆盖首末帧。
- Dense：16 秒×4 FPS=64 帧，每两帧（0.5 秒）一个 1/2/3 分。窗口 10 秒平移，非最后一窗丢掉末端 4 秒，使保留标签拥有后见证据；异常标签附连续区间解释，预测/解析失效的窗口可能使用 all-normal 回退并记录为 invalid（部署需审查）。
- Rubric：用户写入每个步骤的可选离散分值（满分总和为 1），模型以 thinking 模式逐步打分；官方实现默认采样 9 次取 L1 medoid，需要显式识别解析失败与样本不足。
- Rewarder 是共享 Qwen3.5-27B 的**混合 SFT**：Outcome、Dense、Rubric 数据不按类别重采样，前两者直接格式化输出、Rubric 训练包含推理。发布版本 Rewarder 未用 GRPO/RL。

### §4.3 / Appendix F、G：初始化、优化和可复现边界

- TempSeg 初始化自 Qwen3.5-9B，SFT 冻结视觉编码器，语言 attention/MLP rank-32 LoRA（α=64）；Captioner 初始化自 Qwen3.5-27B，SFT 更新语言模型和 visual merger；Rewarder 初始化自 Qwen3.5-27B，SFT 更新语言模型、visual merger、visual encoder（视觉编码器使用更小学习率）。
- SFT 基于 NeMo AutoModel，标注模型的 GRPO 使用自定 PyTorch 实现；三个模型均基于监督输出 token CE 训练基线。公开源码仓仅包含推理示例，**没有**上述训练入口。

### Appendix A–C：评测协议的约束

- 任务不可混淆：分段时间阈值 `tIoU`、描述的 reference-oracle 评分、联合匹配 F1、Outcome MAE、OOD Kendall、VOC 进度相关、Dense F1/Recall、Rubric L1 都是不同统计量。
- Reward 专项基线公开论文的指标保留了原协议，和通用 VLM 并非完全匹配的模型/帧数/服务推理预算对照。
- Robometer ID/OOD VOC 衡量归一化进度随时间的相关性，不能等同真机操作成功率；Dense 内部 bench 的失败 recall 高也不直接意味着能做实时安全监测。

### Appendix D：成本、生产吞吐与可检查假设

- 作者用固定 **50 集、4,483.54 秒视频**计算比较质量，FLORA 成本使用另一次部署研究的 660 个时间窗/890 条 caption 的工作量假设，并非该 RL 模型当场实测吞吐。
- 对 8 张 GB300、每张 $10/GPU-hour、假设速率 1.23495 episodes/s，估计每输入视频小时 **$0.7224**，场景区间 **$0.5102–1.2600**；对最高 effort Astra 的已知用量 **$88.4922** 的比值 **122.49×**，约为论文摘要的 **125×**。
- 生产标注 **200,000 视频小时/天/512 GB300** 为作者披露的数据引擎生产运行指标，**不代表**公开 `captioner/infer.py` 样例吞吐。成本模型不含 storage/服务/计费 overhead，生产 GPU-hour 分母未独立核实。

### Appendix H：与相关资料关系

- 将专用视频标注与 Macrodata Refiner、LeRobot 视频处理、通用 VLM API 和已有 RoboReward/Robometer 专用奖励评估区分；方法实现并非单纯把通用模型两阶段提示拼起来。

## 本次知识编译

- [FLORA 唯一 wiki 实体节点](../../wiki/entities/flora-robotics-data-engine.md)
- [官方项目页核验](../sites/flora-nvidia.md)
- [实际开源推理代码核验](../repos/flora-nvlabs.md)
- [灵巧操作数据管线](../../wiki/queries/dexterous-manipulation-data-pipeline.md)
