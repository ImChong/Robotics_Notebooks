---
type: entity
tags: [paper, stanford, realab, manipulation]
status: complete
updated: 2026-08-18
arxiv: "2606.19586"
venue: "CoRL 2025"
code: https://chuerpan.com/1001-demos.github.io/
related:
  - ../methods/diffusion-policy.md
  - ./paper-umi-ft.md
  - ../tasks/manipulation.md
  - ../overview/realab-14-papers-technology-map-2026.md
sources:
  - ../../sources/papers/action_view_augmentation_arxiv_2606_19586.md
  - ../../sources/blogs/wechat_shenlan_realab_14_papers_2026.md
summary: "Action-View Augmentation（CoRL 2025）：单次鱼眼手眼示范→鱼眼 3DGS 场景编辑+轨迹优化→千条增广轨迹；提升 OOD 位姿/避障成功率。"
---

# One Demo Is Worth a Thousand Trajectories（arXiv:2606.19586）

**One Demo Is Worth a Thousand Trajectories**（Chuer Pan, Litian Liang, Dominik Bauer, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Shuran Song；Stanford University; Columbia University; Toyota Research Institute；[arXiv:2606.19586](https://arxiv.org/abs/2606.19586)，[项目页](https://chuerpan.com/1001-demos.github.io/)）— 从单次真实手眼示范重建鱼眼 3DGS 场景，用轨迹优化生成千条物理可行、视角一致的动作–图像对，增广 visuomotor 训练。

## 一句话定义

从单次真实手眼示范重建鱼眼 3DGS 场景，用轨迹优化生成千条物理可行、视角一致的动作–图像对，增广 visuomotor 训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| 3DGS | 3D Gaussian Splatting | 场景重建与渲染 |
| FoV | Field of View | 鱼眼广角视野 |
| OOD | Out-of-Distribution | 初始位姿/障碍物分布外 |
| DP | Diffusion Policy | 典型被增广训练的策略族 |

## 为什么重要

微小初始位姿变化或新障碍就让 visuomotor 策略崩溃；采集千次真机不现实。

## 核心原理（方法）

单次扫描+示范 → 鱼眼适配 3DGS → 轨迹优化生成无碰撞路径 → 多视角鱼眼渲染。

## 实验与评测

仿真与真机多操作任务；同场景与增广障碍场景成功率均提升。

## 结论

离线 action–view 增广是低成本补齐分布空洞的手段，尤其适合便携鱼眼 UMI 类数据。

- 单次示范可扩到 ~1000 轨迹
- 同时增广视角与障碍物布局
- 轨迹优化保证物理可行性
- 鱼眼 3DGS 是核心工程贡献

## 源码运行时序图

**不适用**（截至 2026-08-18：无统一公开可运行代码仓库，或本文为综述/控制器论文以项目页演示为主）。

## 局限与风险

依赖可重建静态场景；动态障碍与严重光照变化仍难。

## 与其他工作对比

相对 sim 随机化，视觉更逼真；相对纯 sim 增广，锚定真实一次示范。

## 关联页面

- [diffusion-policy](../methods/diffusion-policy.md)
- [paper-umi-ft](./paper-umi-ft.md)
- [manipulation](../tasks/manipulation.md)
- [REALab 14 篇技术地图](../overview/realab-14-papers-technology-map-2026.md)

## 参考来源

- [action_view_augmentation_arxiv_2606_19586.md](../../sources/papers/action_view_augmentation_arxiv_2606_19586.md)
- [wechat_shenlan_realab_14_papers_2026.md](../../sources/blogs/wechat_shenlan_realab_14_papers_2026.md)

## 推荐继续阅读

- 论文：<https://arxiv.org/abs/2606.19586>
- 项目页：<https://chuerpan.com/1001-demos.github.io/>
