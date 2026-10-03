# LeTools-Learning（kuavo_learning_studio）

> 来源归档（repo）

- **标题：** LejuRobotics/LeTools-Learning
- **类型：** repo
- **代码：** <https://github.com/LejuRobotics/LeTools-Learning>
- **旧名/描述：** `kuavo_learning_studio`（GitHub description）
- **文档：** <https://www.letools.lejurobot.com/docs.html?type=learning>
- **许可：** **GPL-3.0**（仓库 `LICENSE`）
- **语言：** Python
- **Stars / Forks（入库快照）：** 35 / 11
- **创建：** 2026-05-29 · **默认分支：** `main`
- **钉版本叙事：** Python 3.12 · LeRobot 0.5.2 · ROS Noetic
- **入库日期：** 2026-08-17
- **一句话说明：** 乐聚官方 **模仿学习 / VLA 胶水仓**：rosbag → LeRobot Dataset v3 → 训练 → 仿真/真机评测；对接 Kuavo 4 Pro / 5 / 5W。

## 开源状态（步骤 2.5）

- **已开源、可运行：** `setup_env.sh`、`kuavo_data/CvtRosbag2Lerobot.py`、`kuavo_model/train.py`、`kuavo_deploy/eval.py`、外部模型目录与 AGX Orin 说明均在仓内。
- **边界：**
  - 依赖 CUDA GPU 训练、Docker/NVIDIA Container Toolkit（仿真或容器化 ROS）、以及真机侧 Kuavo SDK/ROS。
  - 上游 LeRobot 经 `third_party/` + `lerobot_patches/` **钉版本打补丁**，勿盲目升级。
  - 外部模型（OpenPI / GR00T N1.7 / Lingbot-VLA / v2）走 `kuavo_server` adapter：**先起模型服务，再 `policy_type: client`**。
  - 无正式 SLA；企业支持走 `lejurobot@lejurobot.com`。

## README 能力表（归纳）

| 能力 | 说明 |
|------|------|
| 数据转换 | Rosbag → **LeRobot Dataset v3**；配置 `configs/data/KuavoRosbag2Lerobot.yaml`（`platform_type` 4pro/5/5w；`eef_type` leju_claw / rq2f85 / qiangnao） |
| LeRobot 内置策略 | ACT、Diffusion（DPT）、Multi-task DiT；VLA：PI0、PI0_FAST、PI0.5、GR00T N1.5、WALL-X、XVLA、SmolVLA |
| 外部模型 | KUAVO-VLA-1.0 / Pi0 / Pi0.5 / GR00T **N1.7** / LingbotVla / **LingbotVla-v2** |
| 部署 | `configs/deploy/deploy.yaml`：`inference_env` sim\|real；`python kuavo_deploy/eval.py` |
| 训练入口 | `python kuavo_model/train.py --policy <name>`；`--launcher python\|accelerate`；`--mode simple\|total` |
| 边缘 | `README_AGX_ORIN.md` + `requirements_agxorin.txt` |

## News（README）

- 2026-08-26：推出 KUAVO-VLA-1.0，定位为基于 Kuavo 本体与工业场景的垂域模型（资源见 [来源归档](../sites/kuavo-vla-1.md)）
- 2026-05-30：LeRobot 0.5.2 内置 10 种模型 + 原版 lingbotvla / pi0 / pi0fast / pi05 / gr00tN1.7
- 2026-06-13：离线推理、异步推理与 **RTC**
- 2026-07-09：LingbotVLA-v2

## KUAVO-VLA-1.0 资源与开放边界

- **项目页：** <https://model.lejurobot.com/kuavo-vla-1/>
- **模型卡：** HF <https://huggingface.co/LejuRobotics/LET-KUAVO-VLA-1.0-models> · ModelScope <https://www.modelscope.cn/models/lejurobot/LET-KUAVO-VLA-1.0-models>
- **数据集卡：** HF <https://huggingface.co/datasets/LejuRobotics/LET-KUAVO-VLA-1.0-Dataset> · ModelScope <https://www.modelscope.cn/datasets/lejurobot/LET-KUAVO-VLA-1.0-Dataset>
- **社区入口：** <https://openlet.openatom.tech/>
- **访问状态：** OpenLET 将模型与配套数据列为申请审核资源；Hugging Face 模型/数据页也要求同意分享联系信息并通过审核后访问文件。
- **许可边界：** 代码仓许可证为 GPL-3.0；这不自动决定权重或数据集许可。HF 数据卡列出 Apache-2.0 元数据，但内容仍需审核访问；模型权重许可以获批后显示条款为准。

## 目录职责

| 路径 | 职责 |
|------|------|
| `kuavo_data/` | rosbag 转换 |
| `kuavo_model/` | 训练入口 + `external_models/` |
| `kuavo_deploy/` | ROS 评测、仿真/真机 |
| `kuavo_server/` | 标准化模型服务 adapter |
| `configs/` | data / train / deploy / platform / accelerate |
| `lerobot_patches/` | 上游兼容补丁 |

## 与其它乐聚仓的关系

| 仓 | 关系 |
|----|------|
| [letools_opensource](letools_opensource.md) | **技能/行为树运行时**，不是本仓；不要把 BT JSON 场景当成 IL 训练入口 |
| [kuavo_data_challenge](https://github.com/LejuRobotics/kuavo_data_challenge) | 赛事/早期 IL 示例；Learning 是更完整的官方训练产品化仓 |
| OpenLET `kuavo-manip-open` | 社区 AtomGit 范例；与本仓同属「Kuavo + LeRobot」叙事，版本与补丁可能不同 |
| [LET-Base-Dataset](../datasets/let-base-dataset.md) | 训练数据源之一（rosbag → 本仓转换） |

## 对 wiki 的映射

- 升格：[wiki/entities/letools.md](../../wiki/entities/letools.md)
- 对照：[wiki/entities/unitree-lerobot.md](../../wiki/entities/unitree-lerobot.md)（宇树官方 LeRobot 改版）
- 格式：[wiki/entities/lerobot.md](../../wiki/entities/lerobot.md)
- 数据：[wiki/entities/let-base-dataset.md](../../wiki/entities/let-base-dataset.md)、[wiki/entities/icra-2026-real-i.md](../../wiki/entities/icra-2026-real-i.md)
