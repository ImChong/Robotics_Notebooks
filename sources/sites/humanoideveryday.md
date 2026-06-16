# Humanoid Everyday

- **标题**: Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation
- **类型**: dataset / research-portal
- **项目页**: https://humanoideveryday.github.io/
- **论文**: Zhao et al., arXiv:[2510.08807](https://arxiv.org/abs/2510.08807) (2025)
- **机构**: USC · Toyota Research Institute
- **收录日期**: 2026-06-16

## 一句话摘要

大规模 **人形机器人真机/遥操** 操作数据集：**260 任务 · 7 大类 · 10.3k 轨迹 · 300 万+ 帧 @30Hz**，含 RGB / 深度 / LiDAR / 触觉与语言标注；覆盖灵巧操作、人–机交互、下肢 loco-manipulation 等，并提供 **云端标准化评测平台**。

## 为何值得保留

- **「日常应用人形」开源数据**：与 AMASS / OMOMO 等 **人类 MoCap** 不同，本集记录的是 **人形机器人执行开放世界操作** 的多模态轨迹，直接服务 manipulation / loco-manipulation 策略学习。
- **任务广度**：七类含 Loco-Manipulation、可变形物体、关节物体、工具使用、高精度操作、人–机交互等，弥补既有数据集偏固定场景或缺下肢运动的问题。
- **评测闭环**：除数据外提供 cloud evaluation，便于跨实验室对比策略而不仅停留在离线指标。

## 公开要点（编译自项目页）

| 字段 | 内容 |
|------|------|
| 规模 | **10.3k** 轨迹 · **3M+** 帧 · **260** 任务 · **7** 大类 |
| 传感 | RGB、深度、LiDAR、触觉 + 自然语言 |
| 采集 | 高效 **人监督遥操** 管线 |
| 分析 | 论文对代表性 policy learning 方法做分任务强弱分析 |
| 评测 | 云端部署策略 → 受控环境反馈 |

## 对 Wiki 的映射

- **wiki/entities/humanoid-everyday-dataset.md**：数据集实体页（归纳级，区别于 Paper Notebooks 待深读占位）。
- **wiki/entities/paper-notebook-humanoid-everyday-a-comprehensive-robotic-datase.md**：索引实体补链至数据集页。
- **wiki/comparisons/humanoid-reference-motion-datasets.md**：与 MoCap / 预重定向参考库对照。
