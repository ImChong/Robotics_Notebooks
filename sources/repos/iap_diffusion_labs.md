# iap-diffusion-labs（MIT 6.S184 / 6.S975 Labs）

> 来源归档

- **标题：** iap-diffusion-labs
- **类型：** repo
- **链接：** <https://github.com/eje24/iap-diffusion-labs>
- **默认分支：** `2026`（IAP 2026）
- **License：** MIT
- **Homepage：** <https://diffusion.csail.mit.edu>
- **入库日期：** 2026-09-21
- **2026 实验核对（2026-11-16）：** 已查看 `labs/` 三份练习 notebook 的关键讲解与实现入口，以及 `solutions/lab_three_complete.ipynb` 的 latent CFG 训练/采样逻辑；未执行代码。
- **一句话说明：** MIT **6.S184 / 6.S975**（IAP 2026）配套 **Jupyter Lab**：ODE/SDE → flow & score matching → DiT+VAE **latent diffusion** 分步实现；含 `2026/solutions/` 参考解法。
- **沉淀到 wiki：** [mit-flow-matching-diffusion-2026](../../wiki/overview/mit-flow-matching-diffusion-2026.md)

---

## 仓库要点

| 维度 | 内容 |
|------|------|
| 语言 | Jupyter Notebook |
| Lab 1 | 随机动力学、Langevin 等数值实验 |
| Lab 2 | 玩具生成模型：**flow matching + score matching** 对照训练 |
| Lab 3 | **VAE 潜空间 + Diffusion Transformer**；完整 latent diffusion 管线 |
| 维护 | 2026 分支含 2026-01 系数 bugfix changelog |

## 直接运行入口与源码边界

| Lab | 练习骨架 | 官方完成版 |
|-----|---------|-----------|
| 1 | [lab_one.ipynb](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_one.ipynb) | [lab_one_complete.ipynb](https://github.com/eje24/iap-diffusion-labs/blob/2026/solutions/lab_one_complete.ipynb) |
| 2 | [lab_two.ipynb](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_two.ipynb) | [lab_two_complete.ipynb](https://github.com/eje24/iap-diffusion-labs/blob/2026/solutions/lab_two_complete.ipynb) |
| 3 | [lab_three.ipynb](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_three.ipynb) | [lab_three_complete.ipynb](https://github.com/eje24/iap-diffusion-labs/blob/2026/solutions/lab_three_complete.ipynb) |

`labs/` 以 `NotImplementedError` 留出训练损失和类方法供学习者实现；`solutions/` 才包含补齐逻辑。Lab 3 训练入口是 notebook 中 `MNISTVAETrainer.train(...)` → `LatentCFGTrainer.train(...)`，推理/可视化经 `CFGVectorFieldODE`、`EulerSimulator` 和 `vae.decode`，不应误写成独立 Python CLI 或机器人真机策略。可运行入口已存在，**本次未验证训练成功或指标**。

## 对 wiki 的映射

- [MIT Flow Matching & Diffusion 2026（overview）](../../wiki/overview/mit-flow-matching-diffusion-2026.md)
- [sources/courses/mit_flow_matching_diffusion_2026.md](../courses/mit_flow_matching_diffusion_2026.md)

## 参考来源（原始）

- GitHub：<https://github.com/eje24/iap-diffusion-labs/tree/2026>
- 课程站：<https://diffusion.csail.mit.edu/2026/index.html>
