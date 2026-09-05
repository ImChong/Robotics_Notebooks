# plotly.py

> 来源归档

- **标题：** plotly.py — The interactive graphing library for Python
- **类型：** repo / Python 库
- **组织：** Plotly, Inc.
- **代码：** <https://github.com/plotly/plotly.py>
- **文档：** <https://plotly.com/python/>
- **底层 JS：** <https://github.com/plotly/plotly.js>
- **Stars：** ~18.8k（2026-09-05 核查）
- **入库日期：** 2026-09-05
- **许可证：** MIT（代码）；文档 CC BY 4.0
- **一句话说明：** 基于 **plotly.js** 的声明式 Python 交互图表库：Express 一行 API + Graph Objects 细粒度控制；输出 Jupyter / marimo widget、独立 HTML 或 [Dash](https://dash.plotly.com/) 应用；静态图导出走 **Kaleido**。

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（MIT） |
| **代码** | <https://github.com/plotly/plotly.py> |
| **静态导出** | 可选依赖 [Kaleido](https://github.com/plotly/Kaleido)（`pip install -U kaleido`；需 Chrome/Chromium，`plotly_get_chrome`） |
| **Dash 仪表盘** | 独立产品 [Dash](https://github.com/plotly/dash)（本仓 README 互链，非 plotly.py 子模块） |
| **商业服务** | Plotly 提供 consulting / OEM；**核心库无需账号** |

## 核心能力（README 摘要）

| 层级 | API | 用途 |
|------|-----|------|
| 高层 | `plotly.express` (`px`) | 一行绑定 DataFrame → 柱状/散点/折线/3D 等 |
| 低层 | `plotly.graph_objects` (`go`) | 细粒度 trace、子图、动画、自定义布局 |
| 渲染 | plotly.js | 浏览器端 WebGL/SVG；30+ 图表类型 |
| 输出 | `fig.show()` / `write_html` / Jupyter widget | 交互缩放、悬停、图例切换 |
| 静态 | `fig.write_image` + Kaleido | 论文/报告 PNG/SVG/PDF |

### 安装要点

```bash
pip install plotly
# Jupyter widget：pip install jupyter anywidget
# 静态导出：pip install -U kaleido
```

## 机器人 / ML 工程语境

- **训练期标量曲线**：本库更常做 **事后分析、对比表、论文图**；实时 loss/reward 监控优先 [TensorBoard](../../wiki/entities/tensorboard.md) / [W&B](../../wiki/entities/weights-and-biases.md)。
- **真机 / ROS 时序**：高频关节、IMU、控制环对齐用 [PlotJuggler](../../wiki/entities/plotjuggler.md)；Plotly 适合把 **聚合后的 episode 统计、成功率条形图、3D 轨迹摘要** 写进 notebook 或 HTML 报告。
- **3D / 动画**：末端轨迹、点云投影、embedding 可视化；比 matplotlib 默认 3D 交互性更好，但大点云仍可能卡顿。

## 对 wiki 的映射

- 实体页：**`wiki/entities/plotly.md`**
- 交叉：[tensorboard.md](../../wiki/entities/tensorboard.md)、[plotjuggler.md](../../wiki/entities/plotjuggler.md)、[robot-policy-debug-playbook.md](../../wiki/queries/robot-policy-debug-playbook.md)
