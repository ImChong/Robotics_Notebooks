# ReferTrack: Referring Then Tracking for Embodied Visual Tracking（arXiv:2607.20061）

> 来源归档（ingest）

- **标题：** ReferTrack: Referring Then Tracking for Embodied Visual Tracking
- **类型：** paper / embodied-visual-tracking / vla / referring / navigation / sim2real
- **arXiv：** <https://arxiv.org/abs/2607.20061>（Submitted 2026-07-22；PDF：<https://arxiv.org/pdf/2607.20061>）
- **项目页：** <https://medlartea.github.io/referTrack/> — 归档见 [`sources/sites/medlartea-refertrack.md`](../sites/medlartea-refertrack.md)
- **代码：** <https://github.com/MedlarTea/referTrack> — 归档见 [`sources/repos/refertrack.md`](../repos/refertrack.md)
- **视频：** <https://youtu.be/CP7h-tWWABU>
- **作者：** Hanjing Ye、Tianle Zeng、Jiazhao Zhang、Shaoan Wang、Zibo Zhang、Weisi Situ、Yuchen Zhou、Yonggen Ling*、Hong Zhang*（*通讯）
- **机构：** 南方科技大学 RCV Laboratory；腾讯 Robotics X；北京大学；福田实验室
- **入库日期：** 2026-08-12
- **一句话说明：** 把具身视觉跟踪（EVT）拆成「图像空间索引 bbox referring → 条件化轨迹预测」：Refer-CoT 单 token 选目标，TVBI 把历史选定框注入视觉历史；与 Refer-QA 共训；EVT-Bench 单视角 SOTA，Go2 / G1 真机定性验证。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-12）：** 页脚 / Abstract 给出 Code → [MedlarTea/referTrack](https://github.com/MedlarTea/referTrack)。
- **仓库核查（同日）：** 根目录仅有 `README.md`、`assets/`、`method.pdf`；README TODO 全部未勾选（checkpoints / evaluation、dataset、training code、data engine）。
- **结论：** **宣称将开源 / 占位仓** — 无可辨识训练 / 推理 / 部署入口。wiki「源码运行时序图」标 **不适用**。

## 摘录 1：问题与范式（§1–§2）

- **任务：** Embodied Visual Tracking（EVT）——仅靠机载视觉、按自然语言描述持续跟随特定行人；核心耦合两能力：**目标识别**与**轨迹规划**（保持约 1–3 m 跟随距离且目标在 FoV 内）。
- **痛点：** 近期 EVT VLA（TrackVLA / TrackVLA++ 等）把识别与规划统一进 next-token 策略；TrackVLA++ 的 CoT 落在**抽象空间 latent / 极坐标 token**，难监督、与图像检测弱对齐。
- **主张：** *referring then tracking* — 把识别写成对当前前视检测框的**索引多选**（含 `<NO_EXIST>`），再条件化解码航点；历史选定框经 **TVBI**（temporal-viewpoint-bbox indicator）注入视觉历史。
- **对照：** 相对 NavFoM 的 TVI，TVBI 在时间–视角指示上叠加 bbox 几何；相对 TrackVLA++ 的 spatial-aware CoT，Refer-CoT 直接选图像空间索引。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-refertrack.md`](../../wiki/entities/paper-refertrack.md)；与 [VLN 任务页](../../wiki/tasks/vision-language-navigation.md)、[Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md)、[VLA](../../wiki/methods/vla.md)、[NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) 互链。

## 摘录 2：架构与训练（§3）

- **骨干：** Qwen3-4B；视觉双编码器 SigLIP + DINOv2；细粒度当前帧 64 token + 粗历史 4 token/帧；滑窗长度 \(H\)。
- **候选目录：** 当前帧 YOLO11 + ByteTrack 行人检测 → 按框面积取 top-\(K\)，附加虚拟 `<NO_EXIST>`；每条目为 `<ped_k>` + \(\mathcal{P}_{\text{bbox}}\) 嵌入。
- **两段 LLM：** 先出 Refer-CoT \(E_T^{\text{refer}}\)，再出动作 token → MLP ActionHead 解 \(M\) 个 egocentric 航点 \((x,y,\theta)\)。
- **TVBI：** \(E_{\text{TVBI}}(t)=E_{\text{TVI}}(t)+\mathcal{P}_{\text{bbox}}(b_t)\)；历史帧用 FIFO 选定框队列；**当前帧细 token 仅用 TVI（不注 bbox）**，迫使模型靠历史几何 + 原图做 referring。
- **损失：** \(\mathcal{L}=\alpha\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{refer}}+\mathcal{L}_{\text{text}}\)，\(\alpha=10\)；轨迹 MSE + Refer CE + Refer-QA 文本 CE。
- **数据：** Habitat 3.0 / EVT-Bench 训练切分上自建 oracle 专家轨迹 **1.3M**（STT/AT 各下采样至 330K，DT 全留 640K）；Refer-QA 自 SYNTH-PEDES 合成 **1.3M**，与导航 **1:1** 共训；两阶段 SFT（先冻 LLM 训 vision projector，再全参微调、冻视觉编码器）。实现基于 OpenTrackVLA。

**对 wiki 的映射：** 实体页「流程总览」画 referring→queue→TVBI→waypoints；工程实践写清检测器、滑窗、`<NO_EXIST>` 与部署环（云端 WebSocket ~10.6 Hz）。

## 摘录 3：EVT-Bench 与真机（§4 / Table 1–2）

单前视主评测（SR↑ / TR↑ / CR↓）：

| 方法（单视角） | Size | RL | STT | DT | AT |
|----------------|------|----|-----|----|----|
| TrackVLA++ | 7B | – | 86.0 / 81.0 / 2.10 | 66.5 / 68.8 / 4.71 | 51.2 / 63.4 / 15.9 |
| VLingNav | 7B | ✓ | 88.4 / 81.2 / 2.1 | 67.7 / 73.5 / 5.5 | – |
| **ReferTrack** | **4B** | – | **89.4 / 92.5 / 1.6** | **73.3 / 81.8 / 7.6** | **74.1 / 85.7 / 7.7** |

- 相对最强单视角 TrackVLA++：DT **+6.8 SR / +13.0 TR**；AT **+22.9 SR / +22.3 TR**；识别密集切分上可匹敌或超过若干多相机基线。
- DT 消融：去 Refer-CoT+TVBI → SR 55.7（−17.6）；仅去 TVBI → 70.4（−2.9）；oracle GT bbox TVBI → 81.5（接近专家 85.1）→ **识别仍是 DT 主瓶颈**。
- 真机：Unitree Go2 / G1，单前视 RealSense D455；云端推理平均 **10.6 Hz**（检测 ~12 ms/step）；杂乱障碍与多人干扰下定性跟随成功。

**对 wiki 的映射：** 结论强调「图像空间 referring 可补偿有限相机覆盖、减轻对 RL 精修依赖」；局限写清开源占位、CR 在 DT/AT 非最低、真机为定性。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-refertrack.md`**（含流程总览；源码时序图标不适用）。
- 新建 **`sources/sites/medlartea-refertrack.md`**、**`sources/repos/refertrack.md`**。
- 交叉更新 VLN 任务页、Qwen-RobotNav（同 EVT-Bench 语境）。
