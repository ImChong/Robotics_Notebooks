---
type: entity
title: WalkE3-Dataset（E3 运动 CSV 与 MuJoCo 回放）
tags: [dataset, humanoid, mujoco, csv, motion, jackhan-sdu, walke3]
summary: "以 CSV 记录根位姿、四元数与 21 关节角；脚本将序列转为含体坐标系线角速度与关节速度的 JSON 运动文件，并在 MuJoCo 中按帧回放，附带 Matplotlib 速度曲线。"
updated: 2026-05-16
status: complete
related:
  - ./jackhan-walke3-e3-ecosystem.md
  - ./jackhan-mujoco-walke3-simulation.md
  - ../concepts/motion-retargeting.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/jackhan-walke3-dataset.md
---

# WalkE3-Dataset（E3 运动 CSV 与 MuJoCo 回放）

**定位**：把 **28 列 CSV**（根位置、四元数 x-y-z-w、21 关节）处理为 **55 元/帧** 的 JSON 运动（追加体坐标系根线速度、根角速度、关节速度），并提供 `visualize_txt.py` 在 MuJoCo 中播放。

## 核心机制（工程切片）

- **数值微分**：中间帧用中心差分估计线速度；四元数差分换算角速度；边界帧用前向/后向差分（README「Technical Details」节）。
- **坐标变换**：将世界系速度左乘姿态逆，得到 README 所称 body frame 速度，便于与策略观测约定对齐。
- **帧率**：`original_fps` 与 `target_fps` 需满足整除关系以便下采样。

## 流程总览

```mermaid
flowchart LR
  csv["CSV 原始列<br/>28 维 / 帧"]
  c2t["csv2txt.py / csv2txt_back.py<br/>差分 + 坐标变换"]
  txt["JSON 运动 TXT<br/>55 维 / 帧 + 元数据"]
  vis["visualize_txt.py<br/>MuJoCo 回放"]

  csv --> c2t --> txt --> vis
```

## 常见误区或局限

- **README 许可占位**：仓库正文写「Specify your license here」，二次分发前需确认作者最终选择的许可证。
- **数据不是「通用 MoCap」**：列定义绑定 E3 21-DoF 与特定根姿态约定，换机器人需重映射列语义。

## 与其他页面的关系

- **[生态总览](./jackhan-walke3-e3-ecosystem.md)**：说明本仓在整条工具链中的位置。
- **[Motion Retargeting](../concepts/motion-retargeting.md)**：若要把 CSV 运动迁移到其他骨架比例，需要额外的重定向层（本仓未内置）。

## 参考来源

- [WalkE3-Dataset 仓库归档](../../sources/repos/jackhan-walke3-dataset.md)

## 关联页面

- [JackHan-Sdu WalkE3 / HumanoidE3 工具链生态](./jackhan-walke3-e3-ecosystem.md)
- [Mujoco WalkerE3 手柄仿真](./jackhan-mujoco-walke3-simulation.md)
- [Motion Retargeting](../concepts/motion-retargeting.md)

## 推荐继续阅读

- 上游仓库 README：<https://github.com/JackHan-Sdu/WalkE3-Dataset>
