# GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives

> 官方论文来源归档（深读笔记；非论文全文转载）

- **类型：** paper / visual-grounding / physical-intelligence / robotics
- **英文标题：** GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives
- **中文标题：** GroundingPI：以视觉基元迈向物理智能的视觉定位基础模型
- **论文原文（HTML v1）：** <https://arxiv.org/html/2609.39601v1>
- **论文：** <https://arxiv.org/abs/2609.39601>
- **项目页：** <https://groundingpi.github.io/>
- **代码：** <https://github.com/groundingpi/GroundingPI>
- **模型：** <https://huggingface.co/GroundingPI/GroundingPI>
- **作者：** Qize Yu, Lianrui Fan, Boyu Chen, Jiaqi Liang, Xini Ding, Yue Chen, Zetian Song, Yuran Wang, Yi Zou, Kaixuan Wang, Tianxing Chen, Wenxuan Song, Bohan Zhou, Mingleyang Li, Siqiao Huang, Yuqi Ye, Caigao Jiang, Wei Wei, Ruihai Wu, Hang Zhang, Yixiao Ge, Shuchang Zhou, Shilong Liu, Xianming Liu, Ping Luo, Shiyu Huang
- **机构：** XPeng Inc., Peking University, The University of Hong Kong, UC Berkeley, Princeton University, NUS, Tsinghua University, HKUST (GZ)
- **版本：** arXiv:2609.39601v1，2026-09-30
- **原始入库：** 2026-10-14；**本次复核：** 2026-11-09
- **核对范围：** 官方 arXiv v1 HTML 论文主文第 1–6 节、正文表 1–5，方法、输入输出协议、训练/迁移设置、消融与限制相关附录第 7–11 节、基准详细表与案例附录第 12–14 节；并与作者官网及公开源码文档交叉核对。
- **通读：** 已完成
- **一句话说明：** 专门训练精确视觉 grounding，把物体类别、文本和空间点/框统一为带空间 token 的自回归输出；评估该表示向机器人操作与自动驾驶任务的迁移。

## 原文机制摘录（Section 3 / Appendix 7）

1. **模型与可复用视觉基元。** MoonViT-V2（Kimi K3）视觉编码器 → 可学习多模态投影器 → Qwen3-4B 全注意力语言解码器。图像 I 与查询 P 经编码，逐 token 自回归生成标签、边界框或点。共享空间词表包含 1,000 个原子坐标 token（0–999）；`<|object_ref_start|>` 与 `<|object_ref_end|>` 包住语义标签，`<|box_start|>` 与 `<|box_end|>` 包住几何值。对象缺失返回 `None`。
2. **输入输出契约。** 归一化坐标存成离散 token；box 为 xyxy 四元组，point 为 xy 二元组。OCR 将识别文本和区域位置关联；referring/GUI/layout/visual prompting 复用同一格式。源码 `grounding_pi/protocol.py` 定义任务提示、语法及解析，`grounding_pi/client.py` 明确要求保留特殊 token 及相邻坐标 token。
3. **数据引擎。** 公共数据 + 多教师字段级融合 → 任务特定校验 → 接受标注训练统一 grounding expert → 对不确定标签使用局部裁剪/补充教师证据重标 → 派生 detection、referring、pointing、visual prompting、OCR 等监督；依赖失效的标签会撤回，不能将未经证实的教师输出视为真值。
4. **训练分期。** 第一阶段（Pretrain 1）包含 Stage 1 projector 对齐（冻结视觉与语言骨干）、Stage 2 全模块多模态预训练、Stage 3 全模块图文/视频理解；第二阶段（Pretrain 2/SFT，Stage 4）对齐空间坐标并学习统一协议。之后以 GRPO 做强化后训练，仅更新语言侧、冻结视觉编码器与投影器。
5. **强化训练。** 每 prompt 采样 8 个候选，grounding 奖励分别约束完整覆盖与精确定位/输出格式（权重 0.7/0.3），OCR 使用联合文本与几何质量的复合奖励；GRPO clip ε=0.2，KL β=0.02。论文报告监督阶段使用 128 GPU，RL 阶段 256 GPU，不能把 release 中的单机配置当成原实验训练配额。

## 向物理智能迁移（Section 4 / Appendix 8）

- **RoboTwin 2.0 与 RoboCasa-GR1：** 使用 π 风格 flow matching Action DiT。预训练 backbone 的 8 层归一化深度隐藏态，经投影/重采样形成每层 64 个 1024 维 token，进入 Action DiT 的 cross-attention；机器人状态与含噪动作 chunk 在 action expert 中去噪。固定 16 个原子 attention block（8 cross + 8 self），动作 horizon 16，推理 4 个 Euler steps。训练比较固定动作数据、Action DiT、40,000 步和推理预算，改变预训练 backbone。
- **不同 OOD 口径：** RoboTwin Clean2Random 用 Clean 训练、Randomized 场景测环境鲁棒性；RoboCasa-GR1 分别检查 Unseen Appearance / Combinations / Object Types。GR1 基于头部相机，没有腕部相机。
- **nuScenes：** 当前前视图像 + 最多 7 个 0.5 s 间隔自车历史状态，解码相对位移轨迹并统计开环 L2；这是 driving trajectory adaptation，不等同于实际闭环自动驾驶系统。

## 主要结果与协议（Section 5 / Appendices 9、12）

| 比较 | 论文中的测量与读法 |
|---|---|
| 34 项视觉 grounding 基准，11 类能力 | GroundingPI 总平均 73.68%；对照 GPT-6 Astra 71.54%。总分是先按 11 项能力分组再平均，**不是 34 个表格数的直接平均**。 |
| Dense200 box | GroundingPI 74.53 F1mIoU，Astra 65.04；对密集目标定位的改善明显。 |
| RoboTwin 2.0 Full | GroundingPI 66.20% SR；Clean2Random 17.60%，Rex-Omni 14.10%。 |
| RoboCasa-GR1 Full（100% demonstrations） | GroundingPI 37.75% SR，RynnBrain 39.00%；**并非所有 ID 条件均第一**。 |
| RoboCasa-GR1 Full/ID 的数据效率 | GroundingPI 50% 示范：28.75% SR；对比 RynnBrain 75%：27.75%。只是下游示范效率，不代表总预训练成本更低。 |
| nuScenes | 平均 open-loop L2 = 0.296 m；Qwen3-VL-4B 为 0.301 m，RynnBrain 为 0.308 m。 |
| Grounding scaling | 四个曝光点：Avg 61.69→69.71→72.22→73.68；OOD SR 26.59→30.00→32.06→33.38；但 ID SR 最终从 38.75 降至 37.75，不能称所有指标单调提升。 |
| 数据组成消融 | 去掉 OCR 所在组，RoboCasa ID/OOD 分别降低 5.33/3.63 个百分点；这是**组级、条件性消融**，不能证明 OCR 独立导致提升。 |
| token / 生成成本 | COCO / Dense200 平均 7.6/5.1 tokens per box，相比表 5 的 SEED1.5-VL 148.8/74.5；这是序列化长度对照，**非等硬件推理时延对照**。 |

论文 GroundingPI 自身结果采用 5 个 seed × 每 seed 2 次共 10 次；本地复跑基线通常 3 次；外部来源指标与协议不一定完全一致，缺测能力不能按 0 或臆测填补。

## 消融与设计意义（Section 5.3 / Appendices 9–10）

- 量化坐标相对文本坐标：grounding Avg 71.21→73.68；文本坐标生成速度相对为 0.25×。换视觉编码器、去掉第三阶段 GRPO 均会降低整体分数。
- **密集目标**是下游跨领域迁移的空间核心。OCR 联合组有互补性，但“字符边界促进局部空间感知”仍属待验证机制假说，不应写为已确证因果。
- 原文的 Gated DeltaNet / DeepStack、System-1 / System-2 分工讨论属于解释或未来设计方向；不意味着已有部署好的双系统机器人控制器。

## 限制、开放性与复现边界（Section 11 / 官方仓库）

- 弱项含高 IoU 边界、部分小框定位、TotalText OCR、FSC147 visual prompting 和前沿 GUI grounding；总均分**不能替代逐任务检查**。
- 机器人 benchmark 固定 action-side 协议，但各 backbone 的参数规模、预训练来源和架构仍不同；驾驶仅开环；physical prompting、双系统及实时闭环未独立验证。
- 已开放 grounding VLM 权重、代码、推理、批量标注、训练与评测入口；未提供一个直接输入 GroundingPI checkpoint 就可执行真实机器人动作的通用 policy。源码 `vla/` 的 StarVLA/OpenWAM 比较需要独立训练与 checkpoint，勿混淆。
- 代码自有贡献按 Apache-2.0；模型/第三方材料还需遵守 Kimi K3 等上游许可。
- 没有独立 GPU/机器人实验复现记录；以上数值均是论文作者所报告。

## 对 wiki 的映射

- [GroundingPI 主实体](../../wiki/entities/paper-groundingpi.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
- [VLA](../../wiki/methods/vla.md)
- [官方源码核查](../repos/groundingpi.md)
- [官方项目页](../sites/groundingpi-github-io.md)

## 引用

Yu et al., *GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives*, arXiv:2609.39601v1, 2026.
