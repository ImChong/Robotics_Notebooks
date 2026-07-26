# ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU（arXiv:2607.19191）

> 来源归档（ingest）

- **标题：** ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU
- **缩写 / 框架：** **ABot-World-0**（因果学生部署名：`ABot-World-0-5B-LF`）
- **类型：** paper / tech-report / interactive-video-world-model / real-time-deployment
- **arXiv：** <https://arxiv.org/abs/2607.19191>（Submitted 2026-07-21；PDF：<https://arxiv.org/pdf/2607.19191>）
- **项目页：** <https://amap-cvlab.github.io/ABot-World/> — 归档见 [`sources/sites/abot-world.md`](../sites/abot-world.md)
- **在线 Studio：** <https://abot-world.amap.com>
- **代码：** <https://github.com/amap-cvlab/ABot-World>（Apache-2.0）— 归档见 [`sources/repos/abot-world.md`](../repos/abot-world.md)
- **权重：** Hugging Face [`acvlab/ABot-World-0-5B-LF`](https://huggingface.co/acvlab/ABot-World-0-5B-LF)；ModelScope [`amap_cvlab/ABot-World-0-5B-LF`](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF)
- **作者 / 团队：** ABot-World Team（Fan Jiang 等）；**阿里巴巴高德 AMAP CV Lab**
- **入库日期：** 2026-07-26
- **一句话说明：** 在单卡 NVIDIA RTX 5090 上实现键盘动作条件、长程闭环交互式视频世界模型；多源数据基建 + 双向教师→因果学生蒸馏（含 LongForcing）+ 全栈低比特流式推理，720P 最高约 16 FPS、1.2 s 首帧延迟、峰值显存约 19 GiB。

## 开源状态（步骤 2.5）

核查日：**2026-07-26**（项目页 / GitHub README / HF 卡）。

| 产物 | 状态 |
|------|------|
| 推理代码 + 本地 Gradio / Studio | **已开源**（`pipeline/causal_inference.py`、`scripts/inference.py`、`web_client/`） |
| 因果学生权重 `ABot-World-0-5B-LF`（5B，基座 Wan2.2-TI2V-5B） | **已开源**（HF / ModelScope，Apache-2.0） |
| 双向教师权重 | **待发布**（Roadmap 未勾选） |
| 约 500 h 带动作标注训练数据 | **宣称将开源**（2026-07-10 公告；截至入库日未列下载链接） |

**结论：** **部分开源**——可复现 **推理 / 交互 demo**；完整训练教师与数据集仍待发布。勿写成「训练全栈已开源」。

## 摘录 1：问题与四项贡献（§1）

- **目标：** 可进入、可控制、可持续演化的持久生成世界，而非一段视觉上合理的短视频。
- **四耦合瓶颈：** (1) 带可靠动作监督的广覆盖时序数据；(2) 相机漫游 + 第三人称角色统一控制；(3) 自回归历史漂移；(4) 消费级硬件实时部署。
- **贡献：** (i) 原始键盘动作统一 scene roaming / character control + reference-character memory；(ii) WorldExplorer 多源数据基建与 14 项质检；(iii) 双向→因果渐进蒸馏 + **LongForcing**；(iv) LightVAE / 低比特 DiT / SageAttention2 / Fast-RoPE / 有界 KV 等全栈流式栈。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-abot-world-0.md`](../../wiki/entities/paper-abot-world-0.md)；挂 [Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、上游 [Wan](../../wiki/entities/paper-wan-video.md)。

## 摘录 2：数据基建 WorldExplorer（§3）

- **三源：** AAA 游戏（API 真值动作）+ 仿真（UE / ABot-3DGS，轨迹投影伪动作）+ 互联网视频（位姿估计伪标签）。
- **WorldExplorer：** 导航 agent + 并行多模态采集 + 任务模板 + **训练反馈闭环**（弱项诊断 → 重配采集比例）。
- **质检：** 14 项确定性检查（文件 / 视觉 / 几何 / 游戏状态 / 动作对齐 / 元数据）+ VLM 语义评估；第三人称另抽四面人物缩略图作身份记忆。
- **统一动作：** 每帧 8 维 multi-hot（WASD 移动 + IJKL 旋转），与 VAE 时域压缩对齐后每 4 帧 pack 成 32 维 token。

**对 wiki 的映射：** 强调「数据闭环是系统一等公民」；与具身操纵 WM 的数据故事对照而非混同。

## 摘录 3：训练管线与 LongForcing（§4）

1. **双向教师：** 微调预训练 **Wan2.2**；动作经 Action Control Adapter 在 patchify 阶段加性注入；reference-character memory token 前置、非对称注意力保第三人称身份。
2. **Teacher Forcing：** 因果掩码下用 GT 历史适配自回归。
3. **ODE 蒸馏：** 冻结 Stage-1 因果模型，少步逼近其概率流 ODE 干净端点。
4. **LongForcing：** 在学生自 rollout 分布上做扩展时域教师的 **DMD**，缓解短视域训练与长程闭环分布偏移。
5. **部署包络（RTX 5090，1280×704）：** Base 直接 OOM；SageAttention2+LightVAE+低比特可达最高 **≈15.8 FPS**、峰值 VRAM **≤19.3 GiB**；默认质量向 **FP8**；action-to-first-frame **1.2 s**。

**对 wiki 的映射：** 实体页画数据→教师→蒸馏→流式推理流程图与源码运行时序图。

## 摘录 4：评测要点（§5）

| 设定 | 要点 |
|------|------|
| WorldRoamBench | 相对 Genie 3 / HappyOyster / LingBot-World / HY-World 1.5：**5B** 量级下 Strict Acc. **0.5266**（次优，HappyOyster 0.5317）、Partial Acc. **0.7290**、Traj. **0.6752** 等 |
| LongForcing 消融 | 60 s rollout：相对 Causal-Forcing 风格基线，后半程 HPSv3 更高、饱和/模糊/重复更低 |
| 长程压力 | 小时级 / 日级关键帧仍可辨场景结构与活跃运动；OOD 角色–场景统一键盘控制；涌现接触/足迹/墙体阻挡等物理响应 |

**对 wiki 的映射：** 读「可控性 + 长程稳定 + 单卡部署」三轴，而非单看美学分数。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-abot-world-0.md`**（含流程总览 + 源码运行时序图 + 结论）。
- 新建 **`sources/repos/abot-world.md`**、**`sources/sites/abot-world.md`**。
- 交叉更新 [generative-world-models](../../wiki/methods/generative-world-models.md)、[video-as-simulation](../../wiki/concepts/video-as-simulation.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)、[paper-wan-video](../../wiki/entities/paper-wan-video.md)、同机构 [paper-abot-m05](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md)。
