# Dungeon and Visibility World Generator（arXiv:2610.19523）

> 来源归档（ingest）

- **标题：** Dungeon and Visibility World Generator
- **缩写 / 框架：** 可复现的 2D / 3D 地牢环境生成器，附带视线（line-of-sight）可见性引擎
- **类型：** paper / world-generation / visibility / exploration / coverage / rust / python
- **arXiv：** <https://arxiv.org/abs/2610.19523>（Submitted 2026-10-14，v1；PDF：<https://arxiv.org/pdf/2610.19523>）
- **代码：** <https://github.com/EdwinMeriaux/Dungeon_and_Visibility_World_Generator>（仓库公开，已通过 2026-10-15 只读浅克隆核查；README 与 `scripts/run_pipeline.py` 可读）
- **项目页：** 未列出
- **作者：** Edwin Mériaux、Antonio Loría、Gregory Dudek
- **机构：** 未核实（arXiv 摘要页未列出机构）
- **入库日期：** 2026-10-15
- **来源文章：** 微信「具身站 31 篇论文」摘要，见 [sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（文章一句话：房间、通道和障碍可控；输出连续几何与离散占据表示；Rust 引擎查询视线；网格分辨率会改变可见性）
- **一句话说明：** 提供六类几何构造族的 2D / 3D 地牢生成、连续与离散两种可见性计算，以及可复现的实验输出流程。

## 开源状态（步骤 2.5，2026-10-15）

- **论文声明：** 摘要称其为开源框架；arXiv 页面未直接列代码链接，但 README 与仓库内容明确给出 `Dungeon_and_Visibility_World_Generator` 仓库。
- **仓库核查：** 公开仓库，含 `python/wgv`（生成、采样、可见性、验证、渲染）、`rust/visibility_engine`（视线引擎）、`configs/`（示例配置）、`scripts/run_pipeline.py`（入口）、`tests/`。仓库根目录未见 LICENSE 文件（未核实许可证）。
- **结论：** **已开源**（代码与运行入口齐全，许可证未核实）。

## 摘录 1：运行入口（仓库 README）

- 环境：Python 3.11 + Rust 工具链；`pip install -r requirements-lock.txt`；`cargo build --locked --release`。
- 运行：`python scripts/run_pipeline.py --config configs/showcase/2d_polygonal_1.json --output outputs/... --stage all`。
- 阶段：`generate`（几何与占据）、`visibility`（连续与离散可见性）、`validate`、`render`；已有感知结果受保护，实验需使用新的输出目录。

## 摘录 2：方法要点（论文摘要与 README）

- 构造族共六类：散布房间、房间簇、洞穴、蜿蜒通道、分支、不规则回路。几何类别：2D 正交 / 多边形 / 曲线，3D 正交 / 多面体 / 曲线。
- 连续自由空间由生成器得到；离散占据网格为保守近似（2D 单元需完全位于多边形内；3D 体素需满足整体间隙证书）。
- 视线判定：观察者到目标的线段与连续墙体或被占据网格单元求交；离散目标为全部自由单元 / 体素中心。
- 论文分析了几何类别与分辨率对可见性及生成时间的影响。

## 对 wiki 的映射

- 升格页面：[paper-dungeon-visibility-world-generator](../../wiki/entities/paper-dungeon-visibility-world-generator.md)
