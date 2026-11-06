# Frontier3DBench: Benchmarking Coding Agents for 3D Creation across Objects, Scenes, and Interactive Open Worlds

- **论文：** https://lab.einsia.ai/frontier3dbench/Frontier3DBench.pdf
- **项目页：** https://lab.einsia.ai/frontier3dbench/
- **代码与任务：** https://github.com/Einsia/Frontier3D
- **核对日期：** 2026-11-06
- **核对范围：** 官方项目主页与代码仓库 README；论文 PDF 正文尚未提取，源码调用路径尚未逐一核对。
- **通读：** 未完成

## 已核实的项目概况

官方主页和仓库 README 显示，基准包含 145 个任务：L1 对象 65 项、L2 场景 65 项，均使用 Blender；L3 交互式环境 15 项，使用 Unreal Engine。总分按 L1、L2、L3 的 30%、30%、40% 加权。官方仓库提供任务输入、生成与评估代码；模型提交结果、第三方运行时不包含在仓库中。

## 待完成

- 获取并通读论文全文，核验实验设置、结论和局限。
- 阅读实际运行入口及相关源码，核对调用关系。
- 建立符合规范的项目实体节点，补充基于全文的流程图、源码运行时序图和交叉链接。
- 运行 `make ci-preflight` 后，将草稿 PR 转为可审阅状态。
