# Open-Embodied · 官方仓库核对

- **类型：** repo
- **原文：** https://github.com/open-embodied/open-embodied
- **维护方：** GitHub 组织 open-embodied；README 将 DeepOpen 列作支持方。
- **固定版本：** [da44b6600d78220a3f388a54f46aa1fbaefc0ef5](https://github.com/open-embodied/open-embodied/tree/da44b6600d78220a3f388a54f46aa1fbaefc0ef5)，提交于 2026-11-25；该版本包元数据标注 `deephunch-world` 1.5.0。
- **许可证：** 仓库根目录 Apache-2.0；外链项目和自带/下载资产需逐项检查。
- **核对日期：** 2026-11-28
- **核对范围：** 固定提交 README 全文（愿景、资源链接、致谢与对照表）、`pyproject.toml`、`deephunch/__init__.py` 与场景构建和步进、Franka 刚体抓取示例、子模块配置及根 LICENSE；另核对 README 指向的独立 DeepHunch 仓库首页。范围限于本次请求的主仓及其示例路径，不以第三方项目或产品页面替代源码。
- **通读：** 已完成

| 官方入口 | 可核验范围 |
|---|---|
| [README](https://github.com/open-embodied/open-embodied/blob/da44b6600d78220a3f388a54f46aa1fbaefc0ef5/README.md) | 计划按硬件抽象、操作系统、中间件和应用分层；硬件、数据、开发环境和基准多以外链提供入口。规划不等于发布可统一部署的软硬件闭环。 |
| [包元数据](https://github.com/open-embodied/open-embodied/blob/da44b6600d78220a3f388a54f46aa1fbaefc0ef5/pyproject.toml)与[初始化模块](https://github.com/open-embodied/open-embodied/blob/da44b6600d78220a3f388a54f46aa1fbaefc0ef5/deephunch/__init__.py) | 包声明 Python `>=3.12,<3.15`，依赖 Quadrants 1.4.0 和 MuJoCo 等；包初始化还导入 PyTorch。仅能确认源码要求，未验证 PyPI 对应该提交。 |
| [Franka 示例](https://github.com/open-embodied/open-embodied/blob/da44b6600d78220a3f388a54f46aa1fbaefc0ef5/examples/rigid/franka_cube.py)与[Scene](https://github.com/open-embodied/open-embodied/blob/da44b6600d78220a3f388a54f46aa1fbaefc0ef5/deephunch/engine/scene.py) | 初始化 CPU 场景，加载 Panda 模型和方块，设置指端摩擦，调用逆运动学与关节位置控制并逐步推进仿真。 |
| [独立 DeepHunch 仓](https://github.com/open-embodied/deephunch) | 主仓 README 另链到这个子项目；此档仅核对主仓内嵌的代码快照，不以外链仓版本替换。 |

查看代码、场景与依赖是静态核验；未安装依赖、启动仿真、连接真机或复现项目对照表。
