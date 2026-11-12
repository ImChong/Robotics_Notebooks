# Diagrams（mingrammer/diagrams）— Python Diagram as Code

> 来源归档（官方仓库 / 源码，核对日期 2026-11-12）

- **标题：** Diagrams — Diagram as Code for prototyping cloud system architectures
- **类型：** repo / Python library / cloud architecture visualization
- **维护者：** mingrammer 及开源贡献者
- **代码：** https://github.com/mingrammer/diagrams
- **官网：** https://diagrams.mingrammer.com/（另见 [官网归档](../sites/diagrams-mingrammer-com.md)）
- **许可证：** MIT（仓库 LICENSE）
- **版本快照：** `pyproject.toml` 为 0.25.1；`requires-python = "~=3.9"`；版本及依赖应以安装时为准
- **热度快照：** 2026-11-12 GitHub 页面约 42.8k stars（动态数据）
- **通读：** 已完成
- **核对范围：** 官方 README 全文、`pyproject.toml`、`diagrams/__init__.py` 中 Diagram/Cluster/Node/Edge 的图上下文、节点连线及渲染实现、`diagrams/cli.py` 的执行入口；并对照官网 Installation、Diagram、Clusters、Edges、Playground 文档。
- **项目身份：** `mingrammer/diagrams`；非 Mermaid.js、GitDiagram 或 Archify
- **开源状态：** 已开源；库源码、图标资源、CLI、示例和测试均在官方仓库；不提供云资源创建功能
- **沉淀到 wiki：** [Diagrams 独立实体节点](../../wiki/entities/mingrammer-diagrams.md)

## 一手资料核验

| 问题 | 官方依据 | 核验结论 |
|---|---|---|
| 如何绘图 | README + Diagram/Node/Edge/Cluster 类 | Python 实例化节点，运算符连接关系，Graphviz 渲染 |
| 支持范围 | README Providers + `diagrams/` 下模块 | AWS/Azure/GCP/K8s/阿里云/OCI 等及 OnPrem/SaaS/Programming/Generic；“17 家云”不准确 |
| 能否版本管理 | README | Python 描述文件可以放在 Git 中 review/diff；生成的 PNG 本身不是文本 diff |
| 是否操作云 | README NOTE | 只画示意图，不控制资源，也不生成 CloudFormation/Terraform |
| 安装与依赖 | README + pyproject.toml + Installation | Python >=3.9（当前仓库），需要系统 Graphviz 与 Python `diagrams` 包 |
| 是否跨平台 | Installation | macOS、Windows 均给出 Graphviz 安装指令；Linux 同样可安装 Graphviz 后使用 |
| 导出能力 | Diagram 文档与源码 `Diagram.__outformats` | png、jpg、svg、pdf、dot；底层 Graphviz 支持 |
| Playground | 官网 Playground 文档 | 在线 Pyodide + WebAssembly Graphviz，可浏览器出图；不是本地 Python 命令的前置条件 |

**文档差异：** 安装文档页面仍写 Python 3.7+，但当前 README、包元数据均写 Python 3.9+；本次以可安装包的 `pyproject.toml` 为准。官网 Edges 页的部分历史叙述可能不随源码默认值同步；以当前 `Diagram.__init__` 默认 `curvestyle="spline"` 为准。

## 源码结构与关键运行路径

- `pyproject.toml`：Python 包依赖 `graphviz`、`jinja2`；CLI 命令 `diagrams = "diagrams.cli:main"`。
- `diagrams/cli.py`：解析文件路径，读取每个 Python 文件并执行（`exec`）；因此不要用 CLI 运行不可信脚本。
- `diagrams/__init__.py::Diagram`：构造 `graphviz.Digraph`，维护图属性与上下文；上下文结束调用 `render()`。
- `diagrams/__init__.py::Node`：按 provider 节点类选择图标并将节点写入当前 Diagram/Cluster。
- `diagrams/__init__.py::Edge` 与 `Node.__rshift__`：记录有向连接；`Diagram.connect()` 将边交给 `Digraph.edge`。
- `diagrams/__init__.py::Cluster`：嵌套 subgraph 组织边界；最终并入 Diagram。
- `resources/`：预置服务与组件图标；`diagrams/aws`、`diagrams/k8s` 等提供 Python 节点类。

## 工程边界与提醒

1. 架构组件、方向与关联边由作者显式写入；不是扫描云账号或自动从 IaC 还原架构的服务。
2. 图像由 Graphviz 布局，密集连线可能需要调整方向、cluster 和 Edge 属性。
3. 可在 Git 中 diff 的主要是 `.py` 图描述；不能因此宣称生成图片有语义 diff。
4. 官方 CLI 会执行输入 Python 文件，处理未知代码须经过正常代码审查。
5. 推荐在机器人项目中描述仿真、ROS 2 控制、策略推理服务等**概念拓扑**，不应把示意图误作真实部署或安全验证结果。

## 一手链接

- https://github.com/mingrammer/diagrams
- https://github.com/mingrammer/diagrams/blob/master/diagrams/__init__.py
- https://github.com/mingrammer/diagrams/blob/master/diagrams/cli.py
- https://github.com/mingrammer/diagrams/blob/master/pyproject.toml
- https://diagrams.mingrammer.com/docs/getting-started/installation
- https://diagrams.mingrammer.com/docs/guides/diagram
- https://diagrams.mingrammer.com/docs/guides/cluster
- https://diagrams.mingrammer.com/docs/guides/edge
- https://diagrams.mingrammer.com/docs/getting-started/playground
