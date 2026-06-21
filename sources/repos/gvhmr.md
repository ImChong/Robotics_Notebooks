# GVHMR

> 来源归档

- **标题：** GVHMR（Gravity-View Human Motion Recovery）
- **类型：** repo
- **链接：** https://github.com/zju3dv/GVHMR
- **项目页：** https://zju3dv.github.io/gvhmr/
- **论文：** https://arxiv.org/abs/2409.06662（SIGGRAPH Asia 2024；README 标注 TPAMI 2026）
- **入库日期：** 2026-06-08
- **最后更新：** 2026-06-21
- **一句话说明：** 单目视频 **world-grounded** 人体运动恢复（SMPL）；用 **Gravity-View 坐标** 逐帧估计再变换到世界轨迹，常作为 GMR / 人形重定向上游的「视频→人体轨迹」模块。
- **沉淀到 wiki：** 是 → [`wiki/entities/gvhmr.md`](../../wiki/entities/gvhmr.md)

## 仓库要点（2026-06 复核）

### 快速入口

| 入口 | 说明 |
|------|------|
| `tools/demo/demo.py` | 单视频推理；`-s` 跳过 VO（静态相机） |
| `tools/demo/demo_folder.py` | 文件夹批处理 |
| `tools/train.py` | 训练 / 复现 3DPW·RICH·EMDB 测试 |
| `inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt` | 官方 release 权重 |

### 工程更新（README News）

- **2025-03-08：** 默认改用 **SimpleVO**（替代 DPVO），更高效且与 GVHMR 兼容。
- **2025-03-08：** 新增 `f_mm` 指定全画幅相机焦距（mm）。

### 依赖与致谢

- 致谢 **WHAM**、**4D-Humans**、**ViTPose-Pytorch** 等上游实现。
- 安装见 `docs/INSTALL.md`；提供 Colab 与 HuggingFace Space demo。

## 与重定向链的关系

```
单目视频 → GVHMR（GV 中间表征 → 世界 SMPL 轨迹）→ 可选 HTD-Refine → GMR / PHC / OmniRetarget → 机器人参考
```

- [HTD-Refine](../../wiki/entities/paper-htd-refine-monocular-hmr.md) 论文实验将 GVHMR 作为可插拔初始化。
- [GMR](https://github.com/YanjieZe/GMR) README 声明支持 GVHMR 输入。
- [CRISP-Real2Sim](../../wiki/methods/crisp-real2sim.md) scripts 第 5 步为 GVHMR 人体运动估计。

## 对 wiki 的映射

- 论文摘录：[gvhmr_arxiv_2409_06662.md](../papers/gvhmr_arxiv_2409_06662.md)
- 项目页：[gvhmr-zju3dv-github-io.md](../sites/gvhmr-zju3dv-github-io.md)
- 实体页：[wiki/entities/gvhmr.md](../../wiki/entities/gvhmr.md)
