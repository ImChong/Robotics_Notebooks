# artificial-scientist-lab/Differometor — 可微引力波干涉仪仿真器

> 仓库来源归档（ingest）

- **类型：** repo / differentiable-simulation / gravitational-wave / jax / scientific-instrument-design
- **URL：** <https://github.com/artificial-scientist-lab/Differometor>
- **PyPI：** <https://pypi.org/project/differometor/>
- **许可：** **MIT**
- **项目站：** <https://www.learn2design2026.com/>
- **入库日期：** 2026-09-12
- **一句话说明：** JAX 频域干涉仪仿真器，设计对齐 **Finesse**；支持量子噪声与光机效应；为 Learn2Design-2026 提供纯 JAX、可反传梯度/Hessian 的目标函数，GPU 优化相对 Finesse 数值微分可达数量级加速。

## 维护者整理的结构化入口（摘自 README）

| 主题 | 入口 |
|------|------|
| 安装（CPU） | `pip install differometor` |
| 开发模式 | `git clone` + `pip install -e .` |
| GPU | `pip install --upgrade "jax[cuda13]"` |
| 文档 PDF | `media/documentation.pdf` |
| 评测 / 竞赛接口 | [Differometor-Benchmark](https://github.com/artificial-scientist-lab/Differometor-Benchmark)（`dfbench`） |

## 能力要点

- 准静态用户指定干涉仪配置下的平面波传播。
- 光场调制 / 信号传播、量子噪声、光机效应。
- 与 Finesse 对照的精度验证；优化效率为相对 Finesse+CPU 数值微分的核心卖点。

## 对 wiki 的映射

- [`wiki/entities/paper-designing-physics-experiments-with-ai.md`](../../wiki/entities/paper-designing-physics-experiments-with-ai.md)
- [`sources/repos/artificial_scientist_lab_learn2design_2026.md`](artificial_scientist_lab_learn2design_2026.md)

## 当前提炼状态

- [x] 步骤 2.5：**已开源**（MIT + PyPI）
