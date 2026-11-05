# GradSplat 官方代码

- **仓库：** <https://github.com/Jingong-Chen/GradSplat>
- **论文：** <https://arxiv.org/abs/2610.38951>
- **项目页：** <https://jingong-chen.github.io/GradSplat/>
- **核查日期：** 2026-11-05
- **开放范围：** 推理、评测、演示和权重；未见训练入口。README 声明依赖 CUDA，权重首次下载约 4.9 GB；具体第三方许可须查仓库许可说明。
- **入口：** `demo.py` 加载模型、`load_views`、`GradSplat.infer`、保存 PLY 与渲染视频；`gradsplat/model.py` 的 `forward` 调用 `refinement.py` 的 `appearance_gradient` 与 `shape_gradient` 和学习更新器；细化内部用 `torch.enable_grad()` 求梯度。
- **关联：** [论文归档](../papers/gradsplat_2610_38951.md) · [项目页归档](../sites/gradsplat.md) · [实体](../../wiki/entities/paper-gradsplat.md)
