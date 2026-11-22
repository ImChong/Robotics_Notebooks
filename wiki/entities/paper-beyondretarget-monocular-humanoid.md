---
type: entity
tags:
  - paper
  - humanoid
  - motion-retargeting
  - monocular-video
  - teleoperation
  - real-time-control
  - nju
status: complete
project_id: beyondretarget
original:
  - ../../sources/papers/beyondretarget_arxiv_2609_29850.md
  - ../../sources/sites/beyondretarget-plus.md
  - ../../sources/repos/beyondretarget-plus.md
updated: 2026-11-22
arxiv: "2609.29850"
venue: "arXiv 2026"
related:
  - ../overview/embodied-research-12-papers-technology-map.md
  - ../concepts/motion-retargeting-pipeline.md
  - ../methods/motion-retargeting-gmr.md
  - ../methods/neural-motion-retargeting-nmr.md
  - ../tasks/teleoperation.md
  - ./nvidia-sonic.md
  - ./unitree-g1.md
  - ./paper-hrl-stack-01-retargeting-matters.md
sources:
  - ../../sources/papers/beyondretarget_arxiv_2609_29850.md
  - ../../sources/sites/beyondretarget-github-io.md
  - ../../sources/repos/beyondretarget.md
  - ../../sources/sites/beyondretarget-plus.md
  - ../../sources/repos/beyondretarget-plus.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md
summary: "BeyondRetarget（arXiv:2609.29850）base 以单目视频直接预测多机参考动作（仿真 SR 95.86%、约 193 ms）；Plus 2026-11-22 开源冻结 TRAM/VIMO 的移动相机与全局轨迹离线数据引擎，性能数字不得混用。"
---

# BeyondRetarget（单目视频 → 可执行人形 motion）

**BeyondRetarget**（*Learning Executable Humanoid Motions Directly from Monocular Video*，[arXiv:2609.29850](https://arxiv.org/abs/2609.29850)，[项目页](https://bear-ty.github.io/Beyondretarget_page/)）来自 **南京大学** 与 **江苏移动 / 中国移动紫金创新研究院** 等：跳过推理期 **显式 SMPL 人体中间态**，从单目 RGB 直接学 **面向机器人的隐式共享 motion 特征**，解码到 **八款人形** 的根轨迹与关节，并以 **接触感知后处理** 提升时序一致性与物理可信度；支持 **~50 FPS 级流式推理** 与 **单目视觉全身遥操作**（需另接 SONIC 等执行栈）。

## 一句话定义

单目人类视频不经「人体重建→重定向」两阶段，而是端到端输出多机种人形可执行 motion，并用接触约束 refine 降低 collapse 与延迟。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HMR | Human Mesh Recovery | 单目人体 mesh/姿态估计（本工作用 HMR2.0 作冻结视觉骨干） |
| IK | Inverse Kinematics | 接触 refine 阶段的约束腿 IK |
| GMR | General Motion Retargeting | 两阶段对照：优化式重定向 |
| NMR | Neural Motion Retargeting | 两阶段对照：学习式重定向 |
| SR | Success Rate | 仿真/执行成功率 |
| TCAM | Temporal Context Aggregation Module | BiGRU + RoPE + cross-window 时序编码 |

## 为什么重要

- **把 retargeting 从「SMPL 后处理」改成「机器人空间直接监督」。** 人与 humanoid 在 DoF、比例与动力学上差异大；两阶段还把 **视频估计误差** 原样灌进重定向且 **无法端到端反传**。
- **多机共享 encoder、轻量换 decoder。** 相对「新机整套重训 NMR」或论文对照里 **GVHMR/WHAM→优化式 retargeting** 报告的 **motion collapse**，共享 $\mathbf{u}_t$ + robot-specific 残差头更利于 **跨形态扩展**（页内八机 RAMPJPE 表）；**不否定** [GMR](../methods/motion-retargeting-gmr.md) 作为离线运动学工具的价值。
- **实时 teleop 证据。** 同 GPU 上 **~192.8 ms median latency** vs GVHMR→GMR **~1369 ms**；项目页演示 **仅单目 1080p@30Hz** 驱动真机（背景 mocap 未开）。
- **与 [Retargeting Matters](./paper-hrl-stack-01-retargeting-matters.md) 同命题不同解：** 仍关心 **可执行参考质量**，但路线是 **绕过显式 human representation** 而非只优化 GMR 目标。

## 流程总览

```mermaid
flowchart LR
  vid["单目 RGB 视频"]
  det["Person detector\n+ HMR2.0 特征"]
  tcam["TCAM\n共享 motion 表征 u_t"]
  dec["Robot-specific decoder\n残差 root + 关节"]
  ca["Contact-aware refine\n滤波 / root / 腿 IK"]
  out["可执行机器人轨迹\n+ foot contact"]
  vid --> det --> tcam --> dec --> ca --> out
```

## 核心机制（详细）

| 模块 | 职责 |
|------|------|
| 冻结视觉 | 人体检测 + **HMR2.0** 提帧级 human-centric 特征（训练侧仍可用 SMPL 数据做 **统一 human–robot 监督**，推理 **不需** SMPL 文件） |
| TCAM | BiGRU、RoPE self-attention、cross-window attention 建模局部与长程上下文 |
| 解码 | 各机器人 **残差** 预测 root 平移、连续 root 旋转、$d_r$ 维关节 |
| Contact-aware | 足端接触/support 识别 → 时序滤波、root 修正、约束 IK |
| 多机 | G1、R1、H1、T1、Tienkung、GR1-T1、GR2-V3、Atlas 等 |

**两阶段对照（项目页主表，固定相机设定）：** BeyondRetarget **RAMPJPE 26.36 mm**、**Simulation SR 95.86%**；GVHMR→GMR **39.46 mm / 86.46%**；WHAM→GMR **45.88 mm / 81.49%** 且 **Jitter/Accel 显著更差**；qualitative 上两阶段在 **双手抱腰/背后/大画圆/深蹲** 等出现 **motion collapse**。

## 评测与结果

- **精度与鲁棒：** 相对 GVHMR/WHAM→NMR/GMR 与 GT→NMR/GMR，报告更低 RAMPJPE、更低 failure rate（>65 mm / >100 mm）与更低 foot slide；仿真 SR 与 sim RAMPJPE 列见项目页表格。
- **八机泛化：** 共享特征 + 各机 decoder；GMR 未支持机型在表中为「—」。
- **实时：** Latency **192.8 ms**、throughput **~50 FPS**、peak VRAM **7.03 GiB**（同 GPU 流式管线，对照 GVHMR→GMR **1369.4 ms**）。
- **真机：** 项目页含单臂大圆、深蹲抬腿、侧 stretch、双手背后等 **真机 vs 两阶段** 视频；teleop 段声明 **仅单目相机**。

## BeyondRetarget Plus（2026-11-22 扩展）

**定位变更：** Plus 是同一 BeyondRetarget 工作的**离线全局轨迹与互联网视频数据引擎**，并非新论文，也不是把 base 的约 193 ms 流式指标直接延伸到移动相机场景。官方目前仍引用原论文 [arXiv:2609.29850v1](https://arxiv.org/abs/2609.29850)，另发布了 [Plus 项目页](https://bear-ty.github.io/BR_plus_web/) 与 [独立 Plus 代码](https://github.com/bear-ty/BeyondRetarget_plus)。

### 与基础版的工程差异

| 维度 | BeyondRetarget（base） | BeyondRetarget Plus |
|---|---|---|
| 目标 | 单目视频到多机动作；低延迟遥操作 | 海量互联网人体视频到离线多机运动数据 |
| 视觉模型 | HMR2 特征、TCAM、robot decoder | **冻结 TRAM/VIMO 视频骨干** + robot adapter / static head |
| 相机与根轨迹 | base 主要面向固定相机/有限位移 | 支持 moving / static / auto 前端，估计相机世界轨迹并转换 world-Z-up 机器人根运动 |
| 后处理 | 论文接触感知 refine | 完整 **C-RC1** 重投影/足端修正；保存 raw、final 与诊断数据 |
| 权重 / 训练 | base 独立 checkpoint 与运行入口 | Plus 多机器人联合 checkpoint、冻结特征缓存和机器人头/静态头训练入口 |
| 性能证据 | 论文 Table 1/4 中 347/362 成功与 192.8 ms（**仅 base**） | 官网提供动态演示与完整代码；暂未公开足以直接对应上述指标的 **Plus 定量对照表** |

**关键区别：** Plus 源码的 `TramRobotAdapter` 使用冻结的 TRAM/VIMO 人体视频模型、内部 SMPL pose/shape 特征条件与机器人 DoF adapter；因此不能把 base 论文的「不依赖显式 SMPL→GMR 两阶段串联」误写成 Plus 内部完全没有人体表征。

### 流程总览（Plus）

```mermaid
flowchart LR
  Video["连续单人单镜头 RGB 视频"] --> Track["YOLO 人体跟踪框"]
  Video --> Joints["YOLO COCO17 关键点"]
  Track --> Camera["SAM 人体 mask + TRAM 相机前端"]
  Camera --> CamTraj["SPEC / DROID-SLAM / ZoeDepth<br/>相机轨迹与图像内参"]
  Video --> VIMO["冻结 TRAM / VIMO 视频骨干"]
  Track --> VIMO
  CamTraj --> VIMO
  VIMO --> RobotHead["机器人 DoF adapter + static/contact head"]
  CamTraj --> World["世界坐标 root / world-Z-up"]
  RobotHead --> Raw["机器人关节 + 原始 root"]
  Raw --> CRC1["C-RC1 重投影 / 足端修正"]
  World --> CRC1
  Joints --> CRC1
  CRC1 --> Output["final_motion.npz + 诊断报告"]
```

此图为官方 README 与 `scripts/` 入口抽象；Plus root motion 的估计仍来自冻结 TRAM/VIMO，不应把 camera trajectory 画成直接覆盖 root motion 的训练监督。

### 源码运行时序图（Plus）

```mermaid
sequenceDiagram
  autonumber
  participant CLI as 用户 / CLI
  participant Track as extract_video_boxes.py
  participant Pose as extract_video_coco17.py
  participant Cam as prepare_robot_video_camera.py
  participant TRAM as lib.camera (TRAM)
  participant Infer as infer_beyondretarget.py
  participant Model as TramRobotAdapter
  participant CRC as postprocess_beyondretarget.py
  participant NPZ as outputs / final_motion.npz
  CLI->>Track: --video --weights --output track.npz
  Track-->>CLI: 原始帧对齐 bbox/valid/frame
  CLI->>Pose: --video --boxes track.npz
  Pose-->>CLI: coco17.npz
  CLI->>Cam: --video --boxes --camera-mode moving
  Cam->>TRAM: SAM masks + calibrate_intrinsics + run_metric_slam
  TRAM-->>Cam: world_cam_R/T + 相机内参
  Cam-->>CLI: camera.npy + enriched boxes.npz
  CLI->>Infer: --cfg --video --boxes --checkpoint --camera-trajectory --keypoints --confirm-single-shot
  Infer->>Model: infer_sequence(帧, bbox, 内参)
  Model-->>Infer: 冻结视频骨干特征 + robot DoF/root/static
  Infer->>Infer: 相机到 world-Z-up root 坐标转换
  Infer->>NPZ: 原始轨迹 sidecar
  Infer->>CRC: process_prediction(C-RC1, 视频/bbox/keypoints)
  CRC-->>Infer: final motion + diagnostics + stages
  Infer->>NPZ: 保存最终轨迹与报告
```

官方训练链：`cache_tram_robot_features.py`（16 帧冻结 VIMO 特征）→ `train_beyondretarget.py`（机器人头）；另有 `generate_static_labels.py` / `build_true_world_static.py` → `train_static_head.py` 静态头训练。实际部署是 **离线管线**，源码图不代表端到端低时延闭环控制。

### 开源、复现和局限

- **代码已开放：** [Plus 官方仓库](https://github.com/bear-ty/BeyondRetarget_plus) 可查看训练、推理、相机预处理、CPU 回归测试；官方 Drive 提供 `beyondretarget_plus_8robots.pth` 与机器人资源，支持 G1/R1/H1、GR1-T1/GR2-V3、Booster T1、Tienkung、Atlas 八机。
- **复现前置：** Linux x86_64、Python 3.10、CUDA PyTorch；额外准备 SPEC/camcalib、TRAM/DROID-SLAM/SAM/ZoeDepth 模型及授权 SMPL、BEDLAM/H36M/3DPW/EMDB 资源。仓库代码为 **CC BY-NC 4.0**，第三方模型与数据各有许可。
- **输入约束：** C-RC1 推理要求至少 **16 帧连续单人单镜头视频**，bbox / COCO17 / 内参须使用原图像素系并按原始帧对齐；不应把中断目标轨迹插值后伪装成真实观测。
- **失败案例：** 低纹理/均匀背景可能使相机运动估计漂移；滑冰、体操等动作的运动模糊或 OOD 也会导致重建失败；项目页并未提供可推广到任意网络视频的质量保证。
- **待发布边界：** 官方网页「改进骨干、训练数据、监督」细节卡片仍是 `Coming soon`；单目 RGB **低时延直接控制**也是下一项工作预告，不能算作 Plus 已实测的实时功能。
- **实验比较：** 原论文 Table 1 的 95.86% 仿真执行成功率及 Table 4 的 192.8 ms 中位延迟均属于 **base 原论文协议**。Plus 暂未公布同口径量化增益，评估扩产价值应自建视频类型 × 相机模式 × 根轨迹误差 × 足滑 × 有效片段率的质检表。

完整复现资料：[Plus 项目页归档](../../sources/sites/beyondretarget-plus.md) · [Plus 源码归档](../../sources/repos/beyondretarget-plus.md)；工程路径参见 [动作重定向纵深路线](../../roadmap/depth-motion-retargeting.md) 和 [具身数据纵深路线](../../roadmap/depth-embodied-data.md)。

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant User as 用户/CLI
  participant Infer as app/infer/infer_video.py
  participant Pre as 检测+HMR2 特征
  participant Net as TCAM+Decoder+Contact
  participant FS as outputs/ 轨迹与中间结果
  User->>Infer: --video --robots --output_dir
  Infer->>Pre: 逐帧 bbox + HMR2 features
  Pre->>Net: 特征序列
  Net->>Net: motion + foot contact 预测
  Net->>Net: contact-aware 后处理
  Net->>FS: inputs/ + predictions/
  Note over User,FS: 批处理可先 scripts/preprocess_videos.py；真机 teleop 另接 SONIC + GR00T ZMQ
```

## 工程实践（含开源状态）

| 项 | 结论 |
|----|------|
| arXiv | <https://arxiv.org/abs/2609.29850> |
| 项目页 | <https://bear-ty.github.io/Beyondretarget_page/> |
| 代码 | **已开源** [bear-ty/BeyondRetarget](https://github.com/bear-ty/BeyondRetarget)；`scripts/setup_rgb2robo.sh` + Google Drive checkpoint |
| Demo | [HF Space bear-ty/BeyondRetarget](https://huggingface.co/spaces/bear-ty/BeyondRetarget) |
| 版本 | **base**：实时 teleop 取向；**Plus（2026-11 已开源）**：移动相机与大尺度全局轨迹的离线数据生成，独立仓库与运行入口 |
| 依赖 | Linux x86_64、CUDA 12.1、Python 3.10、PyTorch 2.3；HMR2/YOLOv8x 权重随 setup 下载 |
| 真机闭环 | README：**SONIC 单独安装**，经本项目 **GR00T ZMQ** 接口接视觉遥操作 |

## 结论

**BeyondRetarget 用「共享 robot-oriented 视觉 motion 特征 + 接触 refine」把单目视频→人形可执行轨迹做成可实时、可扩展的多机 pipeline，核心增益来自消除两阶段信息瓶颈而非把 GMR 调参做到极致。**

1. **主指标：** 项目页报告 **Sim SR 95.86%**、**RAMPJPE 26.36 mm**，且 **failure rate（>65 mm）0%** 相对两阶段 GVHMR/WHAM 管线显著更低。
2. **collapse 是硬差异：** 双手背后/大画圆等场景两阶段 **GT SMPL 仍失败** 的 qualitative 说明瓶颈不只在估计噪声，而在 **human-centric 中间表示**。
3. **部署读法：** **base** 对低延迟遥操作优先；户外/跟拍与离线动作数据生产可测试已开源的 **Plus**，但需独立测全局轨迹误差和外部相机依赖。
4. **工程入口：** 复现从 `infer_video.py` 与 Drive checkpoint 开始；真机 teleop 预算 **SONIC + ZMQ** 集成时间。
5. **与 GMR、NMR 对照：** 仍可把输出轨迹接 **whole-body tracking / IL**；上游质量逻辑与 [Retargeting Matters](./paper-hrl-stack-01-retargeting-matters.md) 一致，但 **生成路径** 不同。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| GVHMR/WHAM → GMR/NMR | 经典 **human mesh 中间态 + 重定向**；BeyondRetarget 在论文表中 **更低 RAMPJPE / 更高 Sim SR**，qualitative 上少 **collapse** |
| [GMR](../methods/motion-retargeting-gmr.md) | GMR 仍是 **运动学层 IK/QP 重定向工具**；BeyondRetarget 批评的是 **串联 SMPL 估计误差** 的两阶段 **管线**，不是 GMR 单独模块 |
| [NMR](../methods/neural-motion-retargeting-nmr.md) | NMR 学习 human→robot 映射但常 **绑定训练机形**；BeyondRetarget 用 **共享视觉 motion 特征 + 轻量 decoder** 扩八机 |
| [HTD-Refine](../entities/paper-htd-refine-monocular-hmr.md) | 在 SMPL 轨迹上 **后处理 refine**；BeyondRetarget **推理期不要 SMPL** |
| [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) | 下游 **WAM 闭环 loco-manip**；BeyondRetarget 解决 **上游单目→参考 motion / teleop** |

## 局限与风险

- 原论文的定量表只支持 **base** 结论；Plus 的泛化与重建提升尚无同协议全面量化对比，不推断真实机器人成功率或延迟。
- 单目遮挡、全局尺度与相机漂移、缺乏复杂场景接触监督仍限制将生成动作直接当作真机安全执行指令。
- Plus 外部版权模型/数据集需要单独许可，开源仓库不等于全部权重/训练数据可自由商用。

## 关联页面

- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)
- [GMR](../methods/motion-retargeting-gmr.md)
- [NMR](../methods/neural-motion-retargeting-nmr.md)
- [Teleoperation](../tasks/teleoperation.md)
- [SONIC](./nvidia-sonic.md)
- [动作重定向纵深路线](../../roadmap/depth-motion-retargeting.md)
- [具身数据纵深路线](../../roadmap/depth-embodied-data.md)

## 参考来源

- [beyondretarget_arxiv_2609_29850.md](../../sources/papers/beyondretarget_arxiv_2609_29850.md)
- [beyondretarget-github-io.md](../../sources/sites/beyondretarget-github-io.md)
- [beyondretarget.md](../../sources/repos/beyondretarget.md)
- [BeyondRetarget Plus 官方项目页](../../sources/sites/beyondretarget-plus.md)
- [BeyondRetarget Plus 完整源码](../../sources/repos/beyondretarget-plus.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.29850)
- [GitHub README 推理入口](https://github.com/bear-ty/BeyondRetarget/blob/main/README.md)
- [Plus 项目演示与研究说明](https://bear-ty.github.io/BR_plus_web/)
- [Plus 训练 / 相机预处理 / 离线推理源码](https://github.com/bear-ty/BeyondRetarget_plus)
- [Retargeting Matters（两阶段质量命题）](./paper-hrl-stack-01-retargeting-matters.md)
