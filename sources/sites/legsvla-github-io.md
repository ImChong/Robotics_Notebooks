# legsvla-github-io

> 来源归档（site）

- **标题：** LEGS — Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World
- **类型：** site / project-page
- **链接：** <https://legsvla.github.io/>
- **关联论文：** arXiv:2606.01458
- **机构：** Stanford University
- **入库日期：** 2026-06-04（初归档）；**2026-07-16** 按项目页刷新摘录
- **一句话说明：** LEGS 项目页：四步管线（手机扫描 → 3DGS+SAM3D 重建 → MuJoCo 程序化生成 → VLA 微调）、两阶段颜色校准与 LEGS-AUG 重渲染叙事；汇总 1,110 次 G1 真机试验、9/9 (backbone×task) 匹配或超越 teleop、以及相对 mesh-only / teleop 的关键消融图。

## 页面公开资源

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://legsvla.github.io/> |
| arXiv | <https://arxiv.org/abs/2606.01458> |
| 代码 | 页面标注 **Coming Soon**（2026-07-16 抓取；**2026-07-20** 再核仍为 Coming Soon） |

## 首页核心主张（项目页摘录）

- **0** teleoperation demos needed；**1,110** real-robot trials on Unitree G1。
- **9/9** experiments（3 VLA backbone × 3 task）match or beat human teleop（LEGS(200) 设定）。
- **~15×** cheaper adaptation to new scenes & objects（相对重采 teleop：~0.1 GPU-hr vs >1.5 operator-hr per condition）。
- 三档递难 pick-and-place × 三骨干：**ψ0、π0.5、GR00T N1.6**。

## 管线四步（Method 区）

1. **Capture** — 手持场景视频 + 每物体一张照片。
2. **Reconstruct** — 3DGS 静态背景 + SAM3D 物体 mesh。
3. **Simulate & Generate** — MuJoCo 物理 ⊕ 校准后 3DGS 渲染；程序化演示，无 teleop / 种子 demo / 人类视频。
4. **Deploy** — 微调 VLA → Unitree G1 **zero-shot** 部署。

## 两阶段颜色校准（项目页强调）

- **Stage 1：** 校准物体 mesh 颜色（per-mesh）。
- **Stage 2：** 同时作用于 mesh 与 3DGS 背景的全局对齐。
- 页面并排展示 **raw 3DGS+mesh → mesh calibrated → mesh+3DGS calibrated → real camera**（木桌橙子→盘、蓝桌苹果→盒两套条件）。

## LEGS-AUG：外观解耦与鲁棒性

- **Motion 与 appearance 独立记录** → 单 episode 可 ~0.1 GPU-hr 重渲染到新背景/物体/提示词。
- **Default-only 策略**：即便 Task 1，换场景/物体/提示词后成功率 **崩溃至 0–1/10**（页面 failure 视频叙事）。
- **LEGS 重渲染后**：在 scene / object / scene+object 三类偏移下 zero-shot **success**（页面 success 视频叙事）。
- **OOD 物体位姿**：Task 3 上将橙子推到训练分布外左右两侧仍可做 **OOD probe**（页面视频）。

## Key Results 与 FAQ（项目页图表）

### 主结果表叙事

- **LEGS(200)** 在全部 task × backbone 上 **best or tied**。
- 同预算 **LEGS(50) beats Teleop(50)**。
- 最难 **objects+scene** 偏移下，**re-rendering wins**。

### (a) Photorealism beats mesh-only

SAM3D-aug(200) vs LEGS-aug(200) 成功率（页面柱状）：

| Task | SAM3D-aug | LEGS-aug |
|------|-----------|----------|
| 1 | 60% | **100%** |
| 2 | 50% | **80%** |
| 3 | 20% | **40%** |

→ 项目页总结：photorealism 带来 **1.6×–3.25×** end-task success（跨三 backbone）。

### (b) Augmentation beats scale

LEGS(200) default-only vs LEGS-aug(50)（页面柱状）：

| Task | LEGS(200) default | LEGS-aug(50) |
|------|-------------------|--------------|
| 1 | 10% | **50%** |
| 2 | 10% | **40%** |
| 3 | 10% | **30%** |

### FAQ 四问（项目页原文归纳）

| # | 问题 | 结论 |
|---|------|------|
| Q1 | Teleop-free 合成能否匹配 human teleop？ | **Yes** — 9/9；Task 3 teleop **0/10** 三 backbone，LEGS 最高 **6/10** |
| Q2 | 提升是否仅来自数据规模？ | **No** — LEGS(50) 仍 ≥ Teleop(50) |
| Q3 | 光真实感是否必要？ | **Yes** — 相对 mesh-only **1.6×–3.25×**；LEGS(200) 全胜 SAM3D(200) |
| Q4 | 新外观适应效率？ | **~15×** 更便宜；最难 shift 下 LEGS-AUG **100/80/40%**（T1–T3），teleop 与未增广 LEGS **0–10%** |

## BibTeX（项目页提供）

```bibtex
@article{kim2026legs,
  title   = {LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World},
  author  = {Kim, Hojune and Chen, Timothy and Sun, Jiankai and Osterberg, Lars W. and Chen, Qianzhong and Wang, Ke and Schwager, Mac},
  journal = {arXiv preprint arXiv:2606.01458},
  year    = {2026}
}
```

## 对 wiki 的映射

- [paper-legs-embodied-gaussian-splatting-vla](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)
- [legs_arxiv_2606_01458.md](../papers/legs_arxiv_2606_01458.md)
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
- [VLA](../../wiki/methods/vla.md)
- [Teleoperation](../../wiki/tasks/teleoperation.md)
