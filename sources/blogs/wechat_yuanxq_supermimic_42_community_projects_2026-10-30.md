# 具身智能研究室：SuperMimic 42 个复现与扩展项目盘点

> 来源归档（blog / 微信公众号）

- **标题：** 去年被BeyondMimic统治，今年被SuperMimic：还是同一个第一作者，论文出来几天，开源项目已经遍地走了……
- **类型：** blog
- **作者：** 具身智能研究室（Yuanxq，微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/-6fckwPa2Bc8-FhMAzP8Tg
- **发表日期：** 2026-10-30
- **入库日期：** 2026-10-30
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），纯文本，正文约 6100 字符
- **原始落盘：** [wechat_yuanxq_supermimic_42_community_projects_2026-10-30.md](../raw/wechat_yuanxq_supermimic_42_community_projects_2026-10-30.md)
- **一句话说明：** 截至 2026-10-29 检索到的 42 个 SuperMimic（UC Berkeley，arXiv 2610.28480）复现与扩展项目清单，按 9 类整理，每项给仓库链接、Star 数与一句说明。
- **使用边界：** 清单是公众号作者的检索结果；Star 数为 2026-10-29 快照，会变化。**我未逐个打开这些仓库核查**，下表说明均转述自文章，不代表对各项目指标、许可证或可运行性的独立验证。文中也未称其中有论文作者官方代码；与 [paper-supermimic](../../wiki/entities/paper-supermimic.md) 中「官方实现尚未发布」的核查口径不冲突。

## 项目清单（转述自文章）


### 基础复现（7）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| supermimic_vibed | 716 | <https://github.com/zita-ch/supermimic_vibed> | 基于 Isaac Lab 3 与 MuJoCo Warp 的 G1 复现，提供 LAFAN1 训练、完整动作评估和渲染工具。 |
| SuperMimic-Reproduced | 315 | <https://github.com/CuboiLeo/SuperMimic-Reproduced> | 基于 mjlab 从论文重新实现，包含 LAFAN1、BONES-SEED 数据流程、多卡训练、中文实验记录和预训练权重。 |
| SuperMimic Mjlab | 8 | <https://github.com/Breezewrf/supermimic_mjlab> | 面向 G1 的 mjlab 复现，公开归一化、参考朝向等消融实验，以及策略和世界模型权重。 |
| supermimic_vibe | 5 | <https://github.com/dhmbb2/supermimic_vibe> | 基于 mjlab 复现 G1 跟踪，包含 PPO-Sim、PPO-WM、想象时域、价值网络和抗扰实验。 |
| Supermimic_mod | 11 | <https://github.com/YanYuan98/Supermimic_mod> | 以 CuboiLeo 版本为基础，调整参考朝向、输入归一化和训练执行效率，附中文实现说明与实验记录。 |
| realmimic | 0 | <https://github.com/kevinhenryzz/realmimic> | 面向单张 RTX 4090 的大动作库复现，公开模型权重、评估工具，并接入 SONIC 部署运行时。 |
| RaraAlu/Supermimic | 5 | <https://github.com/RaraAlu/Supermimic> | 基于 mjlab 的复现，支持 LAFAN1 与原生 50Hz HDF5 动作数据，包含多卡训练和性能优化工具。 |

### 方法改进（5）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| InstantTrack | 52 | <https://github.com/Huy-Quang-Dao/instantTrack> | 结合 GPU 训练循环优化、策略滑动平均和小型世界模型，报告单张 RTX 5090 约 15 分钟完成 LAFAN1 训练目标。 |
| allmimic | 31 | <https://github.com/Shimingyi/allmimic> | 基于 supermimic_vibe，改进上下半身动作表示、世界模型损失权重与动作采样。 |
| Rethinking SuperMimic | 27 | <https://github.com/btx0424/rethinking-supermimic> | 研究目标条件怎样进入 G1 策略网络，对比直接拼接与 FiLM 特征调制，提供实现对照和回放工具。 |
| SuperMimic on LocoMuJoCo | 5 | <https://github.com/Alchedata/supermimic-locomujoco> | 在 LocoMuJoCo 中训练 H1 跟踪器，探索 ODE 世界模型和动作输出防饱和正则。 |
| Standing MBRL vs PPO | 20 | <https://github.com/JackMin7/supermimic_whole_body_tracking> | 在同一 G1 静态站立任务上对比 SuperMimic 风格世界模型学习与 BeyondMimic 的 PPO 训练。 |

### 本体与硬件（8）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| MorphMimic | 152 | <https://github.com/lzyang2000/morphmimic> | 让宇树 G1、智元 X2 Ultra 和加速进化 T2 共享世界模型与策略，支持完整动作、五点目标和遥操作接口。 |
| SuperMimic Retargeted | 9 | <https://github.com/tianshuwu/supermimic-retargeted> | 将 CuboiLeo 版本扩展到 G1、A3 和 X2 Ultra，补充 SOMA 重定向、坐标修正和足部接触检查。 |
| SuperMimic E1 | 5 | <https://github.com/wenconggan/SuperMimic_e1> | 将 Isaac Lab 3 实现适配到 21 自由度 E1，提供原生 MuJoCo 回放与仿真迁移评估。 |
| SuperMimic Oli | 3 | <https://github.com/liuskywalkerjskd/supermimic_oli> | 将 supermimic_vibed 移植到逐际动力 31 自由度 Oli，包含动作重定向、训练和视频导出。 |
| SuperMimic on RTX 5080 | 2 | <https://github.com/rzc156/supermimic-rtx5080> | 面向单张 16GB RTX 5080 的 mjlab 复现，提供 LAFAN1 训练配置、逐动作评估与演示。 |
| SuperMimic on Apple M5 | 1 | <https://github.com/WXyuany/supermimic-m5> | 使用 CPU MuJoCo 与 PyTorch MPS 在 Apple M5 上训练，公开运行流程、模型权重和逐动作结果。 |
| SuperMimic-MuJoCo | 15 | <https://github.com/panhainan2020/SuperMimic-MuJoCo> | 将 supermimic_vibed 移植到独立 MuJoCo 与 mjlab，提供训练、推理权重及 LAFAN1、100STYLE 评估。 |
| SuperMimic Wheel-Legged | 7 | <https://github.com/RadiantForever/supermimic_wheelbipe> | 面向 wheelbipe 轮腿机器人，将参考动作库改为速度指令库，训练腿部与轮子的协同控制。 |

### 感知与地形（4）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| InstinctMJ · SuperMimic | 301 | <https://github.com/project-instinct/InstinctMJ/tree/supermimic> | 将 SuperMimic 接入 InstinctMJ，研究深度感知驱动的 G1 动作跟踪与跑酷。 |
| Go2-SuperMimic | 69 | <https://github.com/Zhengtianjin/go2-supermimic> | 适配宇树 Go2 的追逐、躲避与地形任务，公开平地、障碍和地形策略及评估工具。 |
| SuperMimic Parkour | 12 | <https://github.com/yhx1203/supermimic_parkour> | 结合 PHP 跑酷动作与地形数据，在 mjlab 中训练 G1 五类跑酷技能，提供权重与浏览器回放。 |
| SuperMimic-perceptive | 6 | <https://github.com/mode1223345/SuperMimic-perceptive> | 基于 Isaac Lab 3，让策略、价值网络和世界模型共享预训练的高度图编码，探索地形感知控制。 |

### 操作与灵巧手（4）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| SuperMimic for Robot Manipulation | 39 | <https://github.com/emoPointer/supermimic_manipulation> | 将方法迁移到 ARX5 机械臂，提供 Lemon、BoxInBin、MugHang 三类操作任务、权重与重置状态数据。 |
| Dex SuperMimic | 74 | <https://github.com/CumulusAlpha/Dex-Supermimic> | 面向无极灵巧手的方块重定向与转笔，提供三类任务、预训练权重和可视化工具。 |
| In-Hand Rotation mjlab | 34 | <https://github.com/Msornerrrr/in-hand-rotation-mjlab> | 在 LEAP Hand 上对比世界模型学习与 PPO，覆盖方块旋转和 22 类物体的指尖旋转，发布训练权重。 |
| Humanoid Mimic · SuperMimic | 7 | <https://github.com/shuayyy/humanoid-amp-ppo> | 在 mjlab 的 G1 行走、下蹲和搬箱任务中，对比 SuperMimic 与 PPO-Sim。 |

### 肌肉与交互（3）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| muscle-supermimic | 33 | <https://github.com/kaibohex/muscle-supermimic> | 将方法用于 MS-Human-700 和 MyoFullBody 肌肉骨骼模型，提供可微肌肉计算、动作重定向与全身跟踪。 |
| SuperMimic-PERA | 0 | <https://github.com/shangchaohu718/supermimic-pera> | 基于 UniSuperMimic，围绕 G1 动作跟踪继续探索身体交互与滑板等任务。 |
| SuperMimic Backflip Diagnostics | 2 | <https://github.com/liudi159/supermimic-backflip-5090> | 聚焦单张 RTX 5090 上的 G1 后空翻，记录接触力、身体穿透、腿部对称性和落地稳定性。 |

### 框架集成（6）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| UniSuperMimic | 204 | <https://github.com/Motphys/Uni-SuperMimic> | 在 UniLab 中复现 SuperMimic，提供 LAFAN1、BONES-SEED 数据管线，以及多仿真后端入口和评估工具。 |
| ScaleBFM_Plus · SuperMimic | 19 | <https://github.com/Hxxxz0/ScaleBFM_Plus> | 将世界模型与想象轨迹策略训练加入 ScaleTrack，结合原有 Transformer 控制模型，提供训练、权重和导出流程。 |
| HoloMotion · SuperMimic | 749 | <https://github.com/HorizonRobotics/HoloMotion> | 将 SuperMimic 接入 HoloMotion 的 G1 资产、HDF5 动作数据和训练流程，支持 MuJoCo Warp 与 Isaac PhysX。 |
| TrackerLab SuperMimic | 13 | <https://github.com/Renforce-Dynamics/trackerlab-supermimic> | 基于 Isaac Lab 与 PhysX，提供多机器人重定向、训练、完整动作评估和策略导出流程。 |
| SuperMimic Lab | 2 | <https://github.com/chu-fa/isaaclab-supermimic> | 将算法封装为可复用的 Isaac Lab 学习器与训练器，提供自定义任务接口及智元 A3 示例。 |
| SuperMimicKit | 4 | <https://github.com/OMEGA-i/SuperMimicKit> | 面向 G1、H2 和 A3 的多本体工具包，集成重定向、训练、策略与世界模型导出及 WebGPU 预览。 |

### 真机与部署（2）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| SuperMimic-Repro-Sim2Real | 36 | <https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real> | 南京大学团队基于 CuboiLeo 版本，补充视频转动作、舞蹈训练与 G1 部署，公开真机演示和预训练模型。 |
| SuperMimic G1 Deployment | 8 | <https://github.com/KyrieYoung926/supermimic_deploy> | 为 SuperMimic-Reproduced 策略提供 ONNX Runtime、MuJoCo 与 Unitree SDK2 运行流程，包含仿真和 G1 真机演示。 |

### 相关探索（3）

| 项目 | Star | 仓库 | 文中说明 |
|---|---|---|---|
| FlatWorld · SuperMimic-2D | 43 | <https://github.com/swaggyliu/FlatWorld> | 在二维人形角色中实现共享世界模型与短时反向传播训练，展示行走、跑步、跳跃和倒地起身等技能。 |
| Ref2Dex · Dex Contact MBRL | 暂不可查 | <https://github.com/hitsz-oyx/Ref2Dex-agent> | 社区目录收录的 Inspire 灵巧手接触动力学与短时策略优化实验；本次检索时仓库链接返回 404。 |
| G0.5 · SuperMimic-style Steering | 0 | <https://github.com/azizkhadraoui/g05> | 在 G0.5 的 VLA 实验中借鉴 SuperMimic 思路，利用世界模型和价值梯度训练动作生成的引导策略。 |

## 观察（文章 + 我的推测）

- 文章事实：发布数日内出现了多框架（mjlab、Isaac Lab 3、UniLab、LocoMuJoCo）、多本体（G1、H1、Go2、轮腿、Oli、E1、A3、X2 Ultra）、以及机械臂/灵巧手/肌肉骨骼等迁移。
- 我的推测：Star 最高的几项（HoloMotion、supermimic_vibed、InstinctMJ、UniSuperMimic、MorphMimic）多依托已有社区框架，迁移成本低；这只是从列表看出的倾向，未验证。
- 文末提到的 Ref2Dex 仓库在检索时返回 404，属于「社区目录收录但不可访问」。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [SuperMimic](../../wiki/entities/paper-supermimic.md) | 主沉淀页：补「社区复现与扩展生态」概览 |
| [awesome-supermimic 索引](awesome-supermimic.md) | 同类社区索引（30 项，10-26）；本文为 42 项（10-29） |
| [HoloMotion](../../wiki/entities/holomotion.md) / [MorphMimic](../../wiki/entities/paper-morphmimic.md) / [muscle-supermimic](../../wiki/entities/muscle-supermimic.md) / [Project Instinct](../../wiki/entities/project-instinct.md) / [mjlab](../../wiki/entities/mjlab.md) | 清单中已有 wiki 节点的项目或框架 |
