# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates（arXiv:2409.06662）

> 来源归档（ingest）

- **标题：** World-Grounded Human Motion Recovery via Gravity-View Coordinates
- **简称：** GVHMR
- **类型：** paper / human motion recovery / monocular video / world-grounded / SMPL
- **venue：** SIGGRAPH Asia 2024 Conference Proceedings；GitHub README 标注 **TPAMI 2026**
- **原始链接：**
  - arXiv abs：<https://arxiv.org/abs/2409.06662>
  - 项目页：<https://zju3dv.github.io/gvhmr/>
  - 代码：<https://github.com/zju3dv/GVHMR>
  - Colab：<https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj>
  - HuggingFace Demo：<https://huggingface.co/spaces/LittleFrog/GVHMR>
- **机构：** 浙江大学（State Key Lab of CAD&CG）等
- **作者：** Zehong Shen*, Huaijin Pi*, Yan Xia, Zhi Cen, Sida Peng †, Zechen Hu, Hujun Bao, Ruizhen Hu, Xiaowei Zhou（* equal contribution）
- **入库日期：** 2026-06-21
- **一句话说明：** 用 **Gravity-View（GV）坐标系** 逐帧估计人体姿态，将 GV 表征经相机运动变换回世界坐标，从而在单目长视频上同时获得 **重力对齐的全局轨迹** 与 **相机系 SMPL**，并避免自回归方法的误差累积。

## 摘要级要点

- **问题：** 单目 **world-grounded HMR** 的核心难点是 **世界坐标系定义随序列变化而歧义**；先前工作用 **帧间相对运动自回归** 缓解，但易 **累积误差**。
- **GV 坐标：** 由 **重力方向 + 相机视线方向** 定义，**每帧唯一**、天然重力对齐，降低 image→pose 映射的学习歧义。
- **非自回归：** **逐帧** 估计 GV 姿态与根速度等中间量，再积分/变换到世界轨迹，避免沿重力方向的长程漂移。
- **工程：** 公开代码与 `gvhmr_siga24_release.ckpt`；在 3DPW、RICH、EMDB 等基准评测；推理在 RTX 4090 上可达 **~5 ms/帧** 量级（不含预处理）。

## 核心摘录（面向 wiki 编译）

### 1) Gravity-View 坐标与 world 恢复

- **链接：** [项目页 Method](https://zju3dv.github.io/gvhmr/)；论文 §3.1
- **摘录要点：**
  - GV 系：$z$ 轴对齐重力，$x$ 轴由相机视线在水平面的投影确定（项目页示意图）。
  - 网络输出 GV 系人体朝向、SMPL 系根速度、关节静止概率等 **中间表征**。
  - 给定 **相机相对旋转**（VO / 陀螺仪 / SimpleVO），将中间量变换到 **世界坐标** 形成全局 SMPL 序列。
- **对 wiki 的映射：**
  - [GVHMR](../../wiki/entities/gvhmr.md) — 「核心机制」与 Mermaid 主干流程

### 2) 预处理与多任务头

- **链接：** [GitHub README](https://github.com/zju3dv/GVHMR)；项目页 Pipeline 图
- **摘录要点：**
  - 预处理：bbox 跟踪、2D keypoints（ViTPose 系）、图像特征、**相对相机旋转**。
  - 2025-03 更新：默认 **SimpleVO** 替代 DPVO（更高效、与 GVHMR 更兼容）；静态相机可用 `-s` 跳过 VO。
  - 新增 `f_mm` 选项指定全画幅相机焦距。
- **对 wiki 的映射：**
  - [GVHMR](../../wiki/entities/gvhmr.md) — 工程实现与 demo 入口

### 3) 训练协议

- **链接：** 项目页 Training；README Reproduce
- **摘录要点：**
  - 混合数据：**AMASS、BEDLAM、H36M、3DPW**。
  - 2×RTX 4090，420 epoch，batch 256，约 13 h；release ckpt 即此配置。
  - 训练阶段 **不做** 测试脚本中的后处理，故全局指标与测试脚本略有差异（README 说明，仍可用于与 baseline 对比）。
- **对 wiki 的映射：**
  - [GVHMR](../../wiki/entities/gvhmr.md) — 训练与基准表

### 4) 与 TRAM / WHAM 及后处理生态

- **链接：** HTD-Refine 论文实验；GMR README；CRISP-Real2Sim scripts
- **摘录要点：**
  - **TRAM** 同样做野外单目全局轨迹，但路线不同；**WHAM** 为 acknowledged 相关工作。
  - **HTD-Refine** 将 GVHMR 作为可插拔初始化，在 EMDB-2 / RICH 上 **+HTD-Refine** 显著降 jitter、改善 WA/W-MPJPE。
  - **GMR**、**HY-Motion**、**ETH G1 diffusion**、**CRISP-Real2Sim** 等机器人/数据管线把 GVHMR 列为 **视频→SMPL** 默认或推荐环节。
- **对 wiki 的映射：**
  - [HTD-Refine](../../wiki/entities/paper-htd-refine-monocular-hmr.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)
  - [humanoid-training-data-pipeline](../../wiki/queries/humanoid-training-data-pipeline.md)

## 对 wiki 的映射（汇总）

- [gvhmr.md](../../wiki/entities/gvhmr.md) — 主沉淀页
- 交叉：[paper-htd-refine-monocular-hmr.md](../../wiki/entities/paper-htd-refine-monocular-hmr.md)、[motion-retargeting-gmr.md](../../wiki/methods/motion-retargeting-gmr.md)、[hy-motion-1.md](../../wiki/methods/hy-motion-1.md)

## 引用（项目页 BibTeX）

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```
