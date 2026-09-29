# HumanEgo（arXiv:2605.24934）

> 来源归档（论文 + 项目页 + GitHub，2026-09-29 核查）

- **标题：** HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos
- **类型：** paper
- **arXiv：** [2605.24934](https://arxiv.org/abs/2605.24934)
- **PDF：** <https://arxiv.org/pdf/2605.24934>
- **项目页：** <https://humanego-ai.github.io/>
- **代码：** <https://github.com/TX-Leo/HumanEgo>
- **视频：** <https://youtu.be/pdL46diijuY>
- **数据：** [HF `Leo-TX/HumanEgo`](https://huggingface.co/datasets/Leo-TX/HumanEgo)（dataset）
- **权重：** [HF `Leo-TX/HumanEgo`](https://huggingface.co/Leo-TX/HumanEgo)（checkpoints）
- **入库日期：** 2026-09-29
- **机构（README）：** Zhi (Leo) Wang, Botao He, Kelin Yu, Seungjae Lee, Ruohan Gao, Furong Huang, Yiannis Aloimonos 等
- **一句话说明：** 仅用 **~30 分钟/任务** 的 **人类 egocentric 视频**（默认 **Project Aria + MPS**），通过 **实体级 HOI 表征 + Interaction-Centric Tokens（ICT）** 与 **flow-matching 策略**，在 **无机器人演示数据** 前提下实现 **零样本 human→robot** 部署；论文报告四任务平均 **~92.5%** 成功率。

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| GitHub `TX-Leo/HumanEgo` | **已开源**（2026-06-07 全量 release；README 称 code/dataset/docs live） |
| HF 数据 + checkpoint | **已发布**；gallery 可浏览 122 clips |
| 硬件 | 采集默认 **Project Aria**；部署参考 **RealSense + Trossen 双臂**（`SKIP_HARDWARE=0 setup.sh`）；README 称正扩展 Quest/AVP/iPhone 等 |

## 主张与数据效率（用户/官网归纳）

- 零样本人类→机器人迁移；**无需机器人数据**
- 每任务约 **30 分钟** 人类视频；任何人/时间/地点可采
- 可部署在不同机器人、相机、环境（工程上需实现 `Camera` / `RobotArm` / `Perception` 接口）

## 方法要点（README + 推理环归纳）

1. **预处理** `preprocess.Preprocess`：VRS + MPS（SLAM + hand tracking）→ 每帧 3D 手/物跟踪；用 **SAM 2、Grounding DINO、CoTracker、Orient-Anything V2** 等。
2. **实体级表征：** 将演示提升为 **hand–object interaction 实体**；推理时输出 **ICT**（各手/物体 6DoF entity）。
3. **视觉对齐：** 控制环输入 **embodiment-agnostic 干净图**（真臂 inpaint 掉、虚拟夹爪渲染）+ ICT。
4. **训练** `training.FlowMatchingTrainer`：在预处理数据上 **flow matching**；CFG job 如 `HumanEgo.yaml`。
5. **部署** `inference/run_inference.py`：感知 → 干净图 + ICT → 策略 → EE 轨迹 → 闭环伺服。

## 对 wiki 的映射

- [`wiki/entities/paper-sa-2605-24934-humanego-zero-shot-robot-learning-from-minutes-o.md`](../../wiki/entities/paper-sa-2605-24934-humanego-zero-shot-robot-learning-from-minutes-o.md)
- 归档：[`sources/repos/humanego.md`](../repos/humanego.md)、[`sources/sites/humanego-ai-github-io.md`](../sites/humanego-ai-github-io.md)
