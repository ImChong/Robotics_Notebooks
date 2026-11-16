# Ropedia W0：跨本体运动-操作世界模型（arXiv:2610.26855）

> 来源归档（paper）

- **英文标题：** Ropedia W0: Scaling Loco-Manipulation World Models across Embodiments with Human Experience
- **类型：** paper / world model / loco-manipulation / cross-embodiment
- **arXiv：** <https://arxiv.org/abs/2610.26855>
- **原文 HTML：** <https://arxiv.org/html/2610.26855v1>
- **项目页：** <https://ropedia.github.io/ropedia-w0/>
- **代码：** <https://github.com/Ropedia/ropedia-w0>
- **模型权重：** <https://huggingface.co/ropedia-ai/Ropedia-W0>
- **评测数据：** <https://huggingface.co/datasets/ropedia-ai/Xp-Teleop>
- **提交日期：** 2026-10-19
- **核对日期：** 2026-11-13
- **核对范围：** 通读 arXiv v1 HTML 全文（§1–§7、附录 A–F，重点核对 §3–§6 的数据、模型、训练、评测及 Appendix A–D 的采集、模型参数、评测指标与消融；参考文献按涉及结论检视）；核对项目页、官方 GitHub README/推理入口及 Hugging Face 权重与 Xp-Teleop 数据卡。
- **通读：** 已完成
- **一句话说明：** Ropedia W0 将人类第一视角经验与机器人数据编码为相机轨迹和投影运动学骨架，训练跨本体的运动-操作视频世界模型。

## 数据与标准化（论文 §3 / 附录 A）

- Xp-Data 整合约 **10,000 h 人类视频 + 3,000 h 机器人视频**：来自 **14 个公开人类数据集、13 个公开机器人数据集**和超过 6,000 小时的自采四鱼眼多视角人类数据。
- 按视频画质和动作有效性筛选，自动生成相机轨迹、手部 MANO、机器人 FK/keypoint、任务/子任务/原子动作描述；机器人关键点需要大量人工补标与纠错。
- 通过深度重投影和轨迹平滑性、2D 关键点检测器及视觉语言模型交叉检查；整理成 **81 帧、640×480** 的训练窗口，具有 RGB、相机、骨架、文字等统一字段。
- Xp-Teleop **27 段长时 G1 遥操作**、约 1 h，包含 **543 个原子动作片段、30 类技能**；双操作者分别控制行走和操作，记录 RGB-D、IMU、关节、触觉、相机与图像关键点。保留失败和恢复片段，不只有成功执行。

## 模型机制（论文 §4 / 附录 B）

- 输入首帧 `I0`、文本、逐帧运动控制；输出预测第一视角视频。用相机位姿产生像素对齐的 **Plücker ray 6 通道图**控制视角运动，用人手关键点或机器人 FK+人工纠正的**投影骨架**控制操作。
- 对齐空间的相机与骨架编码经**零初始化投影**加入视频 latent tokens，文本通过 cross-attention。可选手腕位姿以及宽视野骨架（宽视野分支通过独立 cross-attention 输入）。
- Wan2.2 系列视频扩散骨干：**5B dense** 与 **A14B MoE（约 27B 总参数、14B active）**；先用人类数据预训练，再以人类/机器人数据均衡采样继续训练。
- 用 **Causal Forcing → 非对称 DMD → LongLive** 训练 4 步自回归、流式输出的学生；附录 B.2 报告 640×480 单 GPU **21.3 fps**，约为 50 步 teacher 的 **7 倍**（按完整解码吞吐），最长评估 61 s。

## 结果及适用口径（论文 §6 / 附录 C–E）

| 实验 | 论文报告 | 如何解读 |
|------|----------|----------|
| 人类第一视角测试 | EgoSuite + OpenAoE 各 100 clip，5B pretrain PSNR 18.90，A14B pretrain 19.77 | 评估视角/手动作跟随与视频质量，不是真机任务完成 |
| DreamDojo GR-1（仅 1% 目标数据） | In-lab PSNR：DreamDojo 14B = 18.983、W0 5B = 20.854、W0 A14B = 23.028 | 控制目标数据比例；体现数据效率 |
| Xp-Teleop（零目标数据） | A14B midtrain PSNR 16.42、SSIM 0.537、LPIPS 0.259 | 目标数据 zero-shot；**目标机器人本体可能在 mid-training 的别的数据源中见过**，不能写成未见本体 |
| 跨本体集（8 种本体 / 400 窗口） | A14B midtrain PSNR 21.98，5B midtrain 19.83；去骨架为 18.17 / 18.03 | 骨架条件比相机条件的平均贡献更大；相机在运动视角的子集更关键 |
| 世界动作模型 | LIBERO-Plus：FastWAM 50.06% → W0 初始化 56.82% | 仅用 W0 初始化 FastWAM 视频骨干，**不是 W0 本身直接输出机器人关节动作** |
| 策略评估 / 规划 | RoboTwin 2.0 策略得分 Pearson r=0.984；真实 G1 r=0.996；规划 G1 55% vs 随机选 26% | 对给定 checkpoint、数据和任务成立，不是普适可靠性保证 |

- 论文还有 Ψ0、ω-HOME、AgiBot、长视频 rolling 及 WorldArena 的对照；部分 PSNR/手误差存在不同训练阶段或评测设置的取舍，避免将论文的“领先”概括成所有分项指标第一。
- 附录 F 明确局限：目前 61 秒级滚动仍需更长期记忆；共享运动学接口**没有建模直接接触和物体动力学**；扩展到更多本体或任务尚待验证。

## 开放状态复核（2026-11-13）

- 官方 GitHub README 当前已将 **5B inference code、5B pretrained/midtrained 权重、Xp-Teleop 数据**列为已发布；可从 Hugging Face 模型卡读取两个 10GB safetensors 权重文件。
- **A14B 权重和推理、benchmark fine-tuned 权重、评测代码及后训练代码**仍待发布。源码 Apache-2.0；5B 权重 CC BY-NC 4.0；Xp-Teleop 数据 CC BY 4.0；Wan2.2 基座与示例源数据有各自许可。

## 对 wiki 的映射

- [Ropedia W0 实体页](../../wiki/entities/paper-ropedia-w0.md)
- [Ropedia Xperience 2 环境与行为数据](../../wiki/entities/paper-ropedia-xperience-2.md)
- [世界模型与动作接口](../../wiki/entities/unifolm-world-model-action.md)
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
