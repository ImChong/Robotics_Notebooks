# Ostrich: Taking Large Strides Through Stiff Contact in Differentiable Dynamics（arXiv:2609.08800）

> 来源归档（ingest）

- **标题：** Ostrich: Taking Large Strides Through Stiff Contact in Differentiable Dynamics
- **简称：** Ostrich
- **类型：** paper / differentiable-simulation / contact / gpu-simulation
- **arXiv：** <https://arxiv.org/abs/2609.08800>
- **PDF：** <https://arxiv.org/pdf/2609.08800>
- **项目页：** <https://aleskucera.github.io/ostrich/> — 归档见 [`sources/sites/ostrich.md`](../sites/ostrich.md)
- **代码：** <https://github.com/aleskucera/ostrich> — 归档见 [`sources/repos/ostrich.md`](../repos/ostrich.md)
- **机构：** 布拉格捷克理工大学（Czech Technical University in Prague）
- **入库日期：** 2026-09-10
- **一句话说明：** GPU 刚体仿真器：大步长非光滑 Newton 硬接触 + 隐函数定理 O(1) 记忆伴随；相对 MJX / Newton Semi-Implicit 更高精度、更大并行与梯度可靠性。

## 开源状态（步骤 2.5，2026-09-10）

- **结论：** **已开源** — GitHub 含 `uv sync` 安装、Newton/Warp 子模块、`examples/` 与 `experiments/` 演示；需 CUDA 12+ 与 CMake 3.x。

## 核心摘录（面向 wiki 编译）

### 摘录 1：相对基线

- 相对 MuJoCo 真机轨迹：仿真精度在 **50× 更大时间步** 仍保持；梯度从随机初始化收敛，MJX 慢、Newton Semi-Implicit 易停滞。
- 单 24 GB GPU **8192 并行世界**；无 checkpoint 时基线更早 OOM；warm iteration 比 MJX **211×** 快。

**对 wiki 的映射：** [paper-ostrich](../../wiki/entities/paper-ostrich.md)

### 摘录 2：工程特性

- 精确非穿透接触 + 非光滑摩擦；最大坐标；基于 NVIDIA Warp 与 Newton。
- 演示：三角网格地形上 **10 s 梯度轨迹优化**；Helhest / Marv 等滑移转向与硬接触机器人。

**对 wiki 的映射：** [paper-ostrich](../../wiki/entities/paper-ostrich.md)

## 当前提炼状态

- [x] 项目页与 GitHub 核查（2026-09-10）
- [x] wiki 实体页已建
