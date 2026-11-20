# ZERITH 2.5 官方研究与演示页

- **类型：** 项目主页
- **官方博客：** <https://www.zerith.com/blog/zerith-2-5/>
- **官方报告 PDF：** <https://www.zerith.com/blog/zerith-2-5/assets/ZERITH-V2.5-v1.0.pdf>
- **论文：** <https://arxiv.org/abs/2611.23746>
- **代码：** <https://github.com/ZERITH-ZLab/ZERITH-V2.5>
- **模型：** <https://huggingface.co/OpenZerith/ZERITH-2.5-LIBERO-Plus>
- **数据：** <https://huggingface.co/datasets/OpenZerith/Zerith-H1-1K>
- **收录日期：** 2026-11-19
- **官方博客复核：** 2026-11-20
- **核对范围：** 官方研究项目说明、README 的研究页、报告、模型、数据与源码入口；对照仿真和 H1 真机评测及许可说明。
- **通读：** 已完成

## 项目要点

- 研究目标是厘清 **视频预测能力转为动作可学习性的条件**，而不是仅用视频生成质量证明机器人控制性能。
- 展示 H1 的 Bin lining、SKU sorting 与 Bimanual assembly。SKU sorting 为定性演示，不代表有统计成功率。
- 作者把目标域 **S1 视频适配** 和 **S2 视频–动作联合学习** 分开，以权重迁移和噪声目标消融解释收益来源。
- 官方博客和 arXiv 正文报告同示范、同 30k 更新的静态几何三种子均值：**37.64%→56.80% LIBERO-Plus**（配对增益 **19.16±0.26 pp**）。与单次权重迁移消融的 **37.54%→56.70%** 是不同统计口径。
- 推理：**单张 H800、batch=1** 下单次决策平均 **331.25 ms**、P95 **365.30 ms**；约 **3.02 plans/s**，每次执行四个动作步折合 **12.08 action steps/s**，不等于关节控制环频率。
- 适用边界：固定 Wan2.2-TI2V-5B、3k S1 和 LIBERO 系列的受控实验；5k 总更新预算阶段 S1–S2 反而落后，不能泛化为任意预算下收益。

## 开放状态（2026-11-19）

| 资源 | 状态 |
|---|---|
| 仿真训练/推理/评测 | **代码已公开**：[ZERITH-ZLab/ZERITH-V2.5](https://github.com/ZERITH-ZLab/ZERITH-V2.5) |
| LIBERO / LIBERO-Plus / MetaWorld 权重 | **已发布入口，但 HF 需要授权访问**：官方 README 的 Model Zoo |
| H1 checkpoint | **已发布 mid-training backbone**，真机还需验证后的推理 bundle、SDK/IK |
| 数据 | [Zerith-H1-1K](https://huggingface.co/datasets/OpenZerith/Zerith-H1-1K) 可作为入口；不等于报告约 10,000 h 全部训练数据开放 |
| 许可 | 代码 Apache-2.0；ZERITH 自有权重 CC BY-NC 4.0，商用须另行获得许可 |

## 交叉引用

- [论文原文归档](../papers/zerith_v25_arxiv_2611_23746.md)
- [源码归档](../repos/zerith-v25.md)
- [知识库详情](../../wiki/entities/paper-zerith-v25.md)
