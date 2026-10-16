# Contact-UAN 官方源码核查

> 来源归档（repo / 官方实现）

- **仓库：** <https://github.com/nolie-rolie/contact-uan>
- **组织：** nolie-rolie
- **默认分支：** `main`
- **可见性：** public
- **许可证：** Apache-2.0
- **项目页：** <https://contact-uan.csail.mit.edu/>
- **论文：** <https://arxiv.org/abs/2610.18991>
- **项目页归档：** [contact-uan-csail-mit-edu.md](../sites/contact-uan-csail-mit-edu.md)
- **Wiki 主实体：** [Contact-UAN](../../wiki/entities/paper-contact-uan.md)
- **核查日期：** 2026-10-16

## 仓库覆盖范围

仓库基于 MJLab，README 提供硬件 pipeline guide、部署参考、数据格式说明，以及状态估计、UAN、下游策略的实现入口。其记录的路径是：先训练 WTW 风格策略并在 G1 采集数据，再训练六个状态估计器并重标注硬件窗口，训练 UAN，随后在冻结 UAN 的仿真中训练任务策略。

| 用途 | 仓库入口 | 状态 |
|------|----------|------|
| 环境安装 | `uv sync --locked --extra cu128`；README 另列 CPU 离线安装 | 有安装说明；训练需 Linux 与 NVIDIA GPU |
| 硬件采集与重标注 | Hardware Pipeline Guide、`DATASETS.md` | 有操作流程和格式说明；需自行采集 G1 数据 |
| UAN 与下游策略训练 | `src/mjlab`、`scripts`、`motions` | 代码已公开；策略权重与采集数据未随仓库发布 |
| 真机部署 | `deploy`、Deployment Reference | 提供控制器配置和导出参考；须按硬件前置条件准备环境 |

## 开源与复现边界

- **代码已开源，数据与预训练策略未随仓库发布。** 所以可以审阅和运行代码，但不能仅凭 clone 得到论文中使用的真机训练数据或策略 checkpoint。
- README 指明项目建立在 MJLab 之上且保留 `mjlab` Python 包名；复现环境与依赖版本应以仓库锁文件及最新指南为准。
- Wiki 中的时序图概括公开 README 所述训练链，不代表本次执行过完整的 G1 真机复现。
