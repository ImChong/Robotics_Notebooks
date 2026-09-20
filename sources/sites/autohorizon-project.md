# AutoHorizon 官方项目页（hatchetproject.github.io）

> 来源归档

- **标题：** VLA Knows Its Limits — AutoHorizon
- **类型：** site / project-page
- **URL：** <https://hatchetproject.github.io/autohorizon/>
- **关联论文：** <https://arxiv.org/abs/2602.21445>
- **代码：** <https://github.com/hatchetProject/AutoHorizon>（页内链 GitHub；Apache-2.0）
- **Hugging Face：** **无**（页内未列 HF / ModelScope）
- **机构：** 伊利诺伊大学芝加哥分校（UIC）；思科研究（Cisco Research）
- **入库日期：** 2026-09-20

## 步骤 2.5 开源核查（2026-09-20）

| 项 | 结论 |
|----|------|
| 项目页 Code / GitHub | 链至 **hatchetProject/AutoHorizon**，README 含安装与 LIBERO 评测脚本 |
| 权重 | 依赖 OpenPI **`pi05_libero`** checkpoint（`gs://openpi-assets/...`），**非本仓分发** |
| Hugging Face | 页内与仓库均未列官方 HF |
| **判定** | **已开源（方法 + 评测代码）**；基础 VLA 权重走 OpenPI 外部下载 |

## 页面结构归纳

1. **Motivation：** 不同 execution horizon 导致 LIBERO 成功率大幅波动（示例 \(p{=}10\) vs \(50\) 可视化）。
2. **Method：** cross/self-attention 热力图 → anchor 现象 → AutoHorizon 动态 \(e\)。
3. **Results：** π0.5 × LIBERO 全表；RoboTwin 七任务表；仿真与真机视频（交互段 horizon 缩短）。
