# Diagrams — 官方网站与用户文档

> 来源归档（官方项目主页，核对日期 2026-11-12）

- **类型：** site / documentation / online playground
- **官网：** https://diagrams.mingrammer.com/
- **安装：** https://diagrams.mingrammer.com/docs/getting-started/installation
- **图上下文：** https://diagrams.mingrammer.com/docs/guides/diagram
- **Cluster：** https://diagrams.mingrammer.com/docs/guides/cluster
- **Edge：** https://diagrams.mingrammer.com/docs/guides/edge
- **在线试用：** https://diagrams.mingrammer.com/playground/
- **代码：** https://github.com/mingrammer/diagrams（归档：[sources/repos/mingrammer-diagrams.md](../repos/mingrammer-diagrams.md)）
- **通读：** 已完成
- **核对范围：** 官方 Installation、Diagram、Clusters、Edges、Online Playground 五个完整文档页面及首页提供的功能入口、provider 导航；结合仓库 README 和代码交叉核对。
- **源码状态：** 已开源，官网直接指向 mingrammer/diagrams；在线 Playground 是额外编辑器，不等于部署控制台
- **wiki 映射：** [Diagrams](../../wiki/entities/mingrammer-diagrams.md)

## 官方文档要点

- **最简安装：** 先安装系统 Graphviz，再 `pip install diagrams`；运行用户脚本即可出图；官方同时支持 `diagrams diagram.py` CLI。
- **上下文与格式：** `with Diagram(..., show=False)` 生成文件；默认 PNG，可选 JPG、SVG、PDF、DOT；可设置 `filename`、`graph_attr` 等。
- **节点与分组：** Python provider 类构建服务节点，`Cluster` 支持嵌套分组；Node 的 `>>` / `<<` / `-` 等运算符连接关系。
- **样式：** `Edge` 可声明 label/color/style；复杂图可能产生大量连线。
- **Playground：** 文档标明在浏览器中经 Pyodide 运行 diagrams，通过 WebAssembly Graphviz 布局，支持节点检索、自动补全、导出和分享。
- **兼容范围说明：** 官方首页导航含 AWS、Azure、GCP、K8S、IBM、阿里云、OCI 等，也包含 Generic、Programming、SaaS、C4、GIS、OnPrem；这些不等同于“云厂商数量”。
- **版本口径：** 官网安装页称 Python 3.7+，但仓库 README 与 `pyproject.toml` 要求 3.9+；推荐后者。

## 适用范围

适合系统组件/拓扑沟通、文档、审查和架构演进；不改变真实云资源，不产生 CloudFormation 或 Terraform。生产部署事实仍需以实际配置/服务发现/运行指标核对。

## 外部链接

- https://diagrams.mingrammer.com/docs/getting-started/installation
- https://diagrams.mingrammer.com/docs/guides/diagram
- https://diagrams.mingrammer.com/docs/guides/cluster
- https://diagrams.mingrammer.com/docs/guides/edge
- https://diagrams.mingrammer.com/docs/getting-started/playground
