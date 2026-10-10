# MotionSpaceFlow: Representation-Aware Flow Matching in Direct Motion Space（arXiv:2609.34190）

> 来源归档（最近复核：2026-10-10）

- **标题：** MotionSpaceFlow: Representation-Aware Flow Matching in Direct Motion Space
- **作者：** Qing Yu、Kent Fujiwara（LY Corporation）
- **类型：** paper / text-to-motion / flow-matching / motion-space generation
- **日期：** 2026-09-28（arXiv v1）
- **arXiv：** <https://arxiv.org/abs/2609.34190>
- **HTML：** <https://arxiv.org/html/2609.34190v1>
- **PDF：** <https://arxiv.org/pdf/2609.34190>
- **项目页：** <https://yu1ut.com/MSFlow-HP/> — [项目页归档](../sites/motionspaceflow-yu1ut.md)
- **代码：** <https://github.com/lycorp-jp/MSFlow> — [代码仓归档](../repos/msflow-lycorp-jp.md)
- **预训练权重：** <https://huggingface.co/ly-corporation/MSFlow>
- **对应实体：** [MotionSpaceFlow](../../wiki/entities/paper-motionspaceflow.md)
- **一句话说明：** 在 HumanML3D 的 263D 增量特征或全局 XYZ 坐标上直接做流匹配，不经学习式动作编码器/解码器；使用表示感知的 RA-MMDiT 与推理期关节/帧投影控制。

## 研究问题与方法

常见文本到动作模型先把动作压入学习得到的 latent，再从 latent 解码回动作；压缩与解码会限制帧和关节的直接访问。MotionSpaceFlow（MSFlow）直接把原始连续动作张量作为流匹配变量，保持原始时间分辨率，并设计适配不同运动表示的噪声路径和时序注意力。

1. **两类表示。** HumanML3D 的 263D 表示包含根部速度等增量量，累积后得到全局轨迹；全局 XYZ 变体每帧使用 22 个关节的绝对三维坐标，共 66D，便于直接设定空间约束。
2. **Representation-aware noise scaling。** 初始噪声为标准差为 `s` 的高斯张量；论文分析 `s` 如何影响信号沿 flow path 出现的时间与中间分布条件数，主模型使用 `s=5`。
3. **Clean-motion prediction。** RA-MMDiT 预测干净终点动作 `x̂₁`，再换算成 flow velocity；对分母 `1−t` 做下限裁剪以处理终点附近的数值不稳定。
4. **RA-MMDiT。** 68M 参数、8 个 512 维 block；冻结 DistilBERT 输出词级文本特征，经 flow-time-aware Token Refiner 对齐到模型宽度。联合注意力让每帧动作特征读取相关词语。
5. **注意力匹配表示。** 对帧间增量的 263D 表示使用 causal mask；对全局 XYZ 位置使用 bidirectional mask，利用前后帧上下文保持整段骨架的一致性。
6. **推理期空间控制。** XYZ 模型通过 projection sampling 在推理时固定任意关节、帧和坐标轴；控制坐标没有进入训练目标，因此属于无需控制条件训练的推理期控制。

## 论文结果（按各自评测协议）

HumanML3D 的主结果使用 MARDM 的 67D evaluator，10 次随机评测并报告 95% 置信区间：

| 变体 | R-Precision Top-1 / Top-2 / Top-3 | FID ↓ | MM-Dist ↓ | CLIP ↑ | 读法 |
|------|----------------------------------:|------:|----------:|-------:|------|
| MSFlow 263D | 0.571 / 0.764 / 0.853 | 0.046 ± 0.004 | 2.890 ± 0.010 | 0.686 ± 0.000 | 论文表中检索、MM-Dist、CLIP 点估计更强 |
| MSFlow XYZ | 0.566 / 0.759 / 0.849 | **0.038 ± 0.004** | 2.916 ± 0.008 | 0.676 ± 0.001 | HumanML3D FID 最佳；相对 CMDM 的 0.078 降低 51.3% |

SnapMoGen 的 296D 原生表示实验中，MSFlow 的 R-Precision Top-1/2/3 为 **0.910 / 0.969 / 0.984**，高于 CMDM 的 0.831 / 0.926 / 0.958；FID 为 **16.342**，高于 CMDM（14.451）和 MoMask++（15.061），因此不能将其概括为各项指标都领先。其 multimodality 为 12.538，在表中仅低于 MDM。

HumanML3D 空间约束评测中，MSFlow 在 pelvis 控制时的 R-Precision 为 0.821，平均全关节控制时为 0.818；投影约束的轨迹、位置和平均误差为 0。这个零误差指被施加的线性坐标约束，不表示整段运动满足碰撞、接触或动力学约束。

## 适用边界

- 论文实验覆盖 HumanML3D 与 SnapMoGen；不同数据集、骨架和运动领域是否同样受益仍待验证。
- 全分辨率序列比时序压缩 latent 需要处理更多 token，长动作效率可能受限。
- projection sampler 当前针对关节坐标的线性等式约束；不保证碰撞避免、接触、关节限位或物理可行性。
- 该方法生成的是人体运动学序列，不是机器人策略或关节控制器。要用于机器人，仍需重定向、物理检查与下游控制验证。

## 对 wiki 的映射

- [MotionSpaceFlow 项目实体](../../wiki/entities/paper-motionspaceflow.md)
- [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)
- [动作生成纵深路线](../../roadmap/depth-motion-generation.md)
