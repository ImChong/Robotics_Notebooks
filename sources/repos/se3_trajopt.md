# SE3_TrajOpt（upatras-lar/se3_trajopt）

> 来源归档

- **标题：** Whole-Body Trajectory Optimization in the SE(3) Tangent Space
- **类型：** repo
- **来源：** 帕特雷大学 LAR / Archimedes
- **链接：** <https://github.com/upatras-lar/se3_trajopt>
- **许可：** BSD-2-Clause
- **论文：** arXiv:2508.11520；Humanoids 2025 DOI 10.1109/Humanoids65713.2025.11203204
- **项目页：** <https://lar.upatras.gr/projects/ibrics.html>
- **视频：** <https://www.youtube.com/watch?v=zBJSsiUExCw>
- **入库日期：** 2026-08-18
- **一句话说明：** 论文声明的 SE(3) 切空间全身 TO：Pinocchio 解析导数 + cyipopt；示例在 `src/examples/agile_exps/`。
- **沉淀到 wiki：** [`wiki/entities/paper-se3-tangent-to.md`](../../wiki/entities/paper-se3-tangent-to.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | conda 前缀 `./.conda`，Python 3.13；conda-forge：`pinocchio`、`cyipopt`、`meshcat-python`、`example-robot-data`、`matplotlib` |
| PYTHONPATH | 官方 README：`export PYTHONPATH=$(pwd)/src` |
| 简单示例 | `python src/examples/talos_trajopt.py --vis` |
| 敏捷任务 | `src/examples/agile_exps/`（walk / hopscotch / jump / handstand / backflip / sideflip） |
| 下游 | [AHMP](https://github.com/hucebot/ahmp) 用本 TO 做接触发现内层 |

## 与扩展仓的关系（2026-08-18）

[yusongmin1/go2_flip_TO](https://github.com/yusongmin1/go2_flip_TO) **不是** GitHub 元数据上的 fork，但 README 主体同源，并增加：Go2 AMP 50 Hz txt 导出、`PYTHONPATH` 需同时含 `src/nltrajopt` 与 `src`、默认 MUMPS 而非 HSL。跟论文数字优先官方仓；跟 Go2 数据集优先扩展仓。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-se3-tangent-to](../../wiki/entities/paper-se3-tangent-to.md) | 五种浮动基参数化对比 |
| [ahmp](./ahmp.md) | 接触发现外壳 |
| [go2_flip_to](./go2_flip_to.md) | Go2 空翻脚本与 AMP 导出 |
