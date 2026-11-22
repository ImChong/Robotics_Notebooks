---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-11-22
arxiv: "2502.14795"
venue: "arXiv 2025 (2502.14795v2)"
summary: "Humanoid-VLA（西湖大学，arXiv:2502.14795）：语言-动作预对齐 + 参数高效第一视角视频交叉注意力微调 + 无标注视频自监督 QA 增强；接全身跟踪控制器在 Unitree G1 上完成踢球/避障（9/10）与握手/挥手（8–10/10）等 8 项自主视觉交互。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-06-contact-tasks.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
original:
  - ../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md
---

# Humanoid-VLA

**Humanoid-VLA** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 121/161** 篇，归类为 **06 特殊任务、接触规划与视觉闭环**。

## 一句话定义

Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。
- 人形 Loco-Manip 161 篇 **#121/161** · 特殊任务、接触规划与视觉闭环。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 121/161 |
| 分组 | 06 特殊任务、接触规划与视觉闭环 |
| 原文题目 | Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration |
| 机构 | 西湖大学 Milab、浙江大学、Westlake Robotics |
| 发表日期 | 2025年2月21日（arXiv:2502.14795v2） |
| 论文/项目 | <https://arxiv.org/abs/2502.14795> |

## 核心机制（归纳）

### 策展导读要点与三阶段训练

1. **语言–动作预对齐：** 以 Llama3-70B 为基座，各肢体分部 VQ-VAE（codebook 1024）将非第一视角人类动捕轨迹与文本描述预对齐。
2. **第一视角视觉融合微调：** 冻结预对齐 Transformer 层，插入参数高效交叉注意力层（文本作 Query，第一视角图像特征作 Key/Value）融合视觉上下文。
3. **无标注视频自监督增强：** 自动把从无标注视频提取的原始姿态序列转化为条件问答对（描述+时长+关键帧约束），配合 8×H100（216 小时）扩展视觉–动作覆盖面后接 WBC/跟踪执行。

## 评测与指标

- **运动学保真度（Table 3）：** HumanML3D 上 FID 达 **0.467±0.018**（优于 MDM 0.889、T2M-GPT 0.531）；消融证明引入大规模视频运动恢复数据把 FID 从 0.557 降至 0.467（约 -16%）。
- **物理可执行性（IsaacGym Table 2）：** 多条件中难度（D+T）下，全身跟踪 `Empjpe` 达 **31.07 mm**、`PA-MPJPE` **1.18 mm**。
- **Unitree G1 真机视觉交互（Table 5，各 10 次试验）：** Turn to object 10/10、Hold object 9/10、Wave 10/10、Avoid obstacle 9/10、Jump over 9/10、Dance with partner 8/10、Punch obstacle 10/10、Kick a ball 9/10。

## 与其他工作对比

- **HumanML3D 运动生成基线（Table 3）：** Humanoid-VLA 的 FID 为 **0.467±0.018**，优于 **MDM**（`0.889`）与 **T2M-GPT**（`0.531`）；消融显示引入无标注视频恢复运动 QA 对将 FID 从 `0.557` 降至 `0.467`（约 -16%）。
- **vs [LeVERB](./paper-bfm-36-leverb.md)（同属 [06 特殊任务、接触规划与视觉闭环](../overview/loco-manip-161-category-06-contact-tasks.md)）：** LeVERB 冻结 SigLIP 并用残差运动学编码器将视觉-语言指令压缩成低维潜动作变量 `z` 驱动 System 1 策略；Humanoid-VLA 则以 Llama3-70B 与分部 VQ-VAE 做语言-动作预对齐，再冻结主干插接第一视角交叉注意力层并用无标注视频伪标注扩展覆盖面。
- **vs [GentleHumanoid](./paper-gentlehumanoid.md)（同组接触任务路线）：** GentleHumanoid 专注于肩-肘-腕多连杆阻抗参考动力学与 5–15 N 可调安全力阈值以保障物理接触安全；Humanoid-VLA 侧重保留 VLM 语义理解、融合第一视角视觉观测并接全身跟踪控制器完成踢球、避障与视觉交互。

## 结论

**Humanoid-VLA 的核心是将非第一视角大规模动作先验、第一视角交叉注意力注入与全身跟踪控制器串接，在不牺牲已有语言–动作空间的前提下补齐自主物体感知能力。**

- 关键训练杠杆是用参数高效交叉注意力接入视觉（冻结预对齐语言-动作主干）并用无标注视频伪标注扩展语料（FID -16%）。
- 量化实证已补全：HumanML3D FID 0.467、IsaacGym `Empjpe` 31.07 mm、G1 真机 8 项自主感知任务成功率 80%–100%。
- 适用边界：上层输出仍需底层全身跟踪器稳住动态平衡，且依赖 Llama3-70B 级大参数离线训练算力（8×H100 216 h），文中未公开实时开源推理权重大仓。

## 工程实践

- **冻结预对齐层后插接视觉交叉注意力：** 利用已有丰富的第三视角动作文本对打底，再仅微调视觉编码器与交叉注意力适配器，避免小规模第一视角数据破坏全身动作先验。
- **把高吞吐低质视频恢复作「预对齐」，真机精细任务再配高精度动捕微调：** 原文 Table 4 显示低质量恢复数据在加强预对齐的同时需配合高精度动捕微调才能压缩加速度抖动。

## 常见误区或局限

1. **大基座计算成本高：** 原文基于 Llama3-70B 在 8×NVIDIA H100 上训练 216 小时，推理与边缘机载部署对算力及通信带宽要求较高。
2. **自主视觉闭环仍受限于底层 WBC 稳定裕度：** 模型在踢球、避障与交互任务上验证了感知驱动动作生成，但精细接触操作仍未完全覆盖。

## 关联页面

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-06-contact-tasks.md](../overview/loco-manip-161-category-06-contact-tasks.md)
- 原始 source：[loco_manip_161_survey_121_humanoid-vla.md](../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md)

## 参考来源

- [loco_manip_161_survey_121_humanoid-vla.md](../../sources/papers/loco_manip_161_survey_121_humanoid-vla.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
