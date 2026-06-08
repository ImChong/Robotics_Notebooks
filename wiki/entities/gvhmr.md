---
type: entity
tags: [repo, human-pose, monocular-video, motion-retargeting, upstream]
status: complete
updated: 2026-06-08
summary: "GVHMR（zju3dv/GVHMR）从单目视频恢复全局人体运动（SMPL 系），常作为 GMR 与人形重定向管线的上游观测模块。"
related:
  - ../concepts/motion-retargeting.md
  - ../concepts/motion-retargeting-pipeline.md
  - ../methods/motion-retargeting-gmr.md
  - ./paper-htd-refine-monocular-hmr.md
sources:
  - ../../sources/repos/gvhmr.md
---

# GVHMR

**GVHMR**（Gravity-View Human Motion Recovery，<https://github.com/zju3dv/GVHMR>）从 **单目 RGB 视频** 估计 **全局坐标系下的 SMPL 人体运动**，是人形「视频→动作」链路里最常见的 **上游人体轨迹模块** 之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HMR | Human Mesh Recovery | 从图像恢复人体网格/骨架 |
| SMPL | Skinned Multi-Person Linear Model | 参数化人体模型 |
| MoCap | Motion Capture | 重定向下游消费的参考动作 |
| IK | Inverse Kinematics | 视频轨迹进入机器人前的几何映射 |

## 为什么重要

- **重定向不是第一步**：现场只有手机视频时，必须先有 GVHMR（或同类 HMR）才能把像素变成关节/网格序列，再交给 [GMR](../methods/motion-retargeting-gmr.md) 等几何重定向。
- **生态互操作**：[GMR](https://github.com/YanjieZe/GMR) 官方支持 GVHMR 输入；[HTD-Refine](./paper-htd-refine-monocular-hmr.md) 等后处理可在重定向前改善 jitter/脚滑。
- **数据管线常见位**：HY-Motion、ETH 扩散 locomotion 等多篇工作把 GVHMR 列为视频→SMPL 重建环节。

## 在重定向流水线中的位置

```mermaid
flowchart LR
  vid["单目视频"] --> gv["GVHMR\nSMPL 轨迹"]
  gv --> opt["可选：HTD-Refine 等"]
  opt --> rt["GMR / PHC / OmniRetarget"]
  rt --> pol["WBT / RL 跟踪"]
```

## 局限

- 单目深度与接触歧义会带来脚滑、漂移；不宜直接把原始输出当真机指令。
- 与棚拍 MoCap / AMASS 相比，噪声更大，常需物理筛选或 RL 修补层。

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)
- [GMR](../methods/motion-retargeting-gmr.md)
- [HTD-Refine](./paper-htd-refine-monocular-hmr.md)

## 参考来源

- [GVHMR 仓库归档](../../sources/repos/gvhmr.md)

## 推荐继续阅读

- GitHub：<https://github.com/zju3dv/GVHMR>
- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)
