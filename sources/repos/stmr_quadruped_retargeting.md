# STMR 四足时空重定向

> 来源归档

- **论文：** Spatio-Temporal Motion Retargeting for Quadruped Robots（IEEE T-RO 2025；arXiv:2404.11557）
- **类型：** paper + project site（GitHub 子仓曾引用但已不可公开访问）
- **入库日期：** 2026-06-08
- **最后核验：** 2026-06-08
- **一句话说明：** SMR + TMR 将动物/视频关键点转为四足全身可跟踪参考；一手入口为 arXiv 与作者项目页，而非失效的 terry97-guel GitHub 链接。
- **沉淀到 wiki：** 是 → [`wiki/entities/stmr-quadruped-retargeting.md`](../../wiki/entities/stmr-quadruped-retargeting.md)

## 一手链接（HTTP 200，2026-06-08）

| 链接 | 角色 |
|------|------|
| <https://arxiv.org/abs/2404.11557> | 论文预印本 |
| <https://taerimyoon.me/Spatio-Temporal-Motion-Retargeting-for-Quadruped-Robots/> | 作者官方项目页（真机视频、方法说明） |
| <https://terry97-guel.github.io/STMR-RL.github.io/> | 论文正文声明的代码页；301 → 官方项目页 |

## 失效链接（勿作一手入口）

以下 URL 于 2026-06-08 返回 **404**，GitHub API 亦报 **Not Found**：

- `https://github.com/terry97-guel/STMR_RL`
- `https://github.com/terry97-guel/Quadruped_Retargeting`
- `https://github.com/terry97-guel/Quadruped-Motion-Timing`

可能原因：仓库删除、改名或改为非公开；二手 README/博客仍引用旧路径。

## 方法摘录

- **SMR：** 运动学层，从关键点生成全身运动，抑制脚滑/穿地，可处理无全局基座轨迹的输入。
- **TMR：** 动力学层，在时间域用模型基控制搜索可行时序参数。
- **下游：** legged_gym 风格 RL 跟踪；论文报告 Go1、Aliengo、B2 等硬件实验。
