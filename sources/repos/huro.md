# HuRo（Robotizing Human Videos）

> 来源归档

- **标题：** HuRo: Robotizing Human Videos for Scalable VLA Pretraining
- **类型：** repo
- **来源：** RLWRLD × 延世大学（Yonsei University）；Jinho Jeong / Se June Joo 等
- **链接：** <https://github.com/3587jjh/HuRo>
- **项目页：** <https://3587jjh.github.io/HuRo/>
- **论文：** <https://arxiv.org/abs/2609.10706>（CoRL 2026）
- **许可：** Apache-2.0（自有代码）；**不可商用**（第三方依赖限制）
- **入库日期：** 2026-09-22
- **一句话说明：** 发布 **10 阶段** egocentric 人视频机器人化流水线（标注 → PyRoKi 重定向 → Isaac Sim 叠加 → LeRobot V2.0）；默认目标 **ALLEX**；单卡 RTX 5090 约 **8–10×** 片长。
- **沉淀到 wiki：** [`wiki/entities/paper-huro.md`](../../wiki/entities/paper-huro.md)

---

## 核心定位

把 **原始 egocentric `.mp4`**（30 fps、短边 256 px）经估计相机/手/语言 → 动作重定向 → 视觉机器人化，写成 **LeRobot V2.0** episode（观测视频 + 关节 state/action + 语言）。

> **注意：** 发布代码 **不从源数据集读现成标注**，一律从视频估计；若已有标注需按 `examples/README.md` 格式写入 stage N-1 输出。

---

## 仓库入口

| 组件 | 说明 |
|------|------|
| 一键跑通 | `./run_pipeline.sh`（默认 `examples/clips/`） |
| 分阶段 | `pipeline/stage<N>_<phase>_<name>.py --input_dir … --part a/b --no_tqdm` |
| 目标机器人 | `--robot_name allex`（`configs/allex.yaml` + URDF）；stage 8–10 可换 embodiment |
| 读 Parquet 标注 | `python examples/read_parquet.py examples/clips_chunked` |
| 读 LeRobot | `python examples/load_lerobot.py examples/clips_lerobot/allex/192x342` |
| 安装 | `setup/README.md`：Linux + **≥24 GB VRAM** + CUDA 12.8；overlay 需 **RT core** 且驱动 **≤ R580** |

---

## 10 阶段流水线（`pipeline/README.md`）

| # | 脚本 | 后端 | 输出 |
|---|------|------|------|
| 1 | `stage1_annot_intrinsics.py` | DroidCalib / AnyCalib | 相机内参 JSON |
| 2 | `stage2_annot_contact.py` | 100DoH | 手部框 + 首份 Parquet |
| 3 | `stage3_annot_contact_refine.py` | BoT-SORT | 一致 hand-side 轨迹 |
| 4 | `stage4_annot_hand.py` | HAWOR | MANO 3D 手姿 |
| 5 | `stage5_annot_extrinsics.py` | DROID-SLAM + MoGe-2 + GeoCalib | 度量重力对齐相机轨迹 |
| 6 | `stage6_annot_narr.py` | ViTDet-H + Qwen3.5-9B | 操作片段 + 语言指令 |
| 7 | `stage7_annot_inpaint.py` | ViTDet-H + SAM2 + ProPainter | 去臂 inpaint 视频 |
| 8 | `stage8_robot_retarget.py` | PyRoKi IK (JAX) | 机器人关节 / 腕姿 |
| 9 | `stage9_robot_overlay.py` | Isaac Sim | ALLEX 叠加渲染 |
| 10 | `stage10_lerobot_convert.py` | — | LeRobot V2.0 数据集 |

Stage 1–7 与机器人无关；已有 `_chunked` 树可 **只重跑 8–10** 换 embodiment。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-huro](../../wiki/entities/paper-huro.md) | 论文实体、VLA 预训练结论与真机数字 |
| [VLA](../../wiki/methods/vla.md) | 下游预训练 + 少量真机微调范式 |
| [motion-retargeting](../../wiki/concepts/motion-retargeting.md) | Stage 8 PyRoKi 两阶段 IK |
| [HumanNet 语料对照](../../wiki/comparisons/humannet-table1-human-video-corpora.md) | 五源人视频输入覆盖面 |
