# ChemLab-Assets：可仿真化学实验室设备资产（Simulation-Ready Chemistry Laboratory Instruments） — repos 来源归档

- **类型：** repo
- **标题：** ChemLab-Assets：可仿真化学实验室设备资产（Simulation-Ready Chemistry Laboratory Instruments）
- **机构：** 穆罕默德·本·扎耶德人工智能大学（MBZUAI）
- **原文：** https://github.com/mbzuai-nlp/ChemLab-Assets
- **项目主页：** https://mbzuai-nlp.github.io/ChemLab-Assets/
- **代码：** https://github.com/mbzuai-nlp/ChemLab-Assets
- **开放状态：** 代码 MIT；三维资产与数据 CC BY-NC 4.0，商用须单独核对。1.5.0（2026-10）发布。
- **入库日期：** 2026-11-24
- **核对范围：** 官方 README、chemlab 使用文档与通风柜 quickstart、gserve 入口；未逐项测试全部仪器。
- **通读：** 已完成
- **对应知识页：** [chemlab-assets](../../wiki/entities/chemlab-assets.md)

## 资料要点

- 每台设备提供 GLB/URDF/OBJ/Blender 几何、asset.contract.json 和可执行行为模型。契约定义关节、单位、状态可见性、能力参数及错误；机器人可经仿真器操作，语言智能体可经 chemlab 调用相同设备能力。仿真时钟、可观察状态与评估器隐藏真值分离，任务检查器读取执行后的状态。
- 官方 1.5.0 报告 203 套设备、119 家厂商、16 个领域，106 个设备带工作菜单；提供 276 个可自动检查任务。官方展示 Isaac Sim 机器人场景以及命令行交互；Matterix 仿真场景本身不在本次发布范围。
- 模型根据公开手册和文献构建，并非原厂 CAD，未用实体设备标定；行为模型不是任意化学过程的真实精度保证。资产非商业许可与代码 MIT 不可混淆。

## 原文通读补充（2026-11-24）

### 行为契约与调用边界

- 契约将部件绑定至网格和碰撞体，状态槽区分设定值、传感值、推导值及过程真值；每项能力记录 JSON Schema 参数、前置条件、状态变化、错误及 process/host/scene/robot/hardware 验证级别。通风柜示例通过 session.invoke、read、describe 演示角色隔离；gserve 为通用实验室 CLI 服务入口。
- 资产的 GLB、URDF、Blender 与碰撞代理服务于不同环节；CLI 行为模型与仿真器接触执行并非同一验证层。

### 验收与限制

- 1.5.0 发布 203 台设备、119 家厂商、16 类领域、276 项可自动检查任务。README 明确 787 项能力声明但不支持，未声称任何能力通过真机或机器人接触验证。代码 MIT，资产、契约与数据 CC BY-NC 4.0，部分数据受原来源单独条款约束。验收须同时检查几何、行为、权限与实际应用许可。

## 公开资源

- [官方项目页](https://mbzuai-nlp.github.io/ChemLab-Assets/)
- [官方源码](https://github.com/mbzuai-nlp/ChemLab-Assets)
- [资产数据集](https://huggingface.co/datasets/MBZUAI/ChemLab-Assets)
- [3D 浏览器](https://huggingface.co/spaces/MBZUAI/ChemLab-Assets-Explorer)

## 对 wiki 的映射

- [chemlab-assets 独立实体](../../wiki/entities/chemlab-assets.md)
- [关联归档](../sites/chemlab-assets.md)
