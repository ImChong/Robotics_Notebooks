# Everything（HeyPuter/everything）

- **标题：** The Everything Project
- **类型：** repo
- **来源：** HeyPuter
- **链接：** <https://github.com/HeyPuter/everything>
- **项目页：** 仓库 README（仓库元数据未提供独立 homepage）
- **入库日期：** 2026-10-20
- **一句话说明：** 将 Office、协作、创意、工程、生产力、财务、业务、开发工具与游戏等多类应用集中在一个 MIT 许可的 JavaScript monorepo 中，从零构建开源替代品。
- **沉淀到 wiki：** 暂不。项目覆盖面很广，但并非机器人研究或工程工具集；其中 CAD 应用与工程软件有间接关联，暂不足以形成机器人知识页。

## 开源状态核查（2026-10-20）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — GitHub 仓库公开，仓库元数据与 README 标注 MIT License |
| **Stars（GitHub API）** | 3,012（核查日） |
| **主要语言** | JavaScript |
| **创建日期** | 2026-10-13 |
| **代表应用** | Spreadsheet、Documents、Teamchat、Photo Studio、CAD 等 |
| **本地运行** | README 要求 Node.js 22.12+；进入 `src/[app-name]` 后执行 `npm install`、`npm run dev` |
| **许可边界** | 仓库声明 MIT；使用前仍应核对各应用及其依赖中的第三方许可 |

## README 摘要

Everything 以一个仓库维护多个应用，面向常见商业软件提供免费、开源的替代实现。应用按类别收录在 README 中，包含办公、协作、创意、工程、生产力、财务、业务、开发工具与游戏。

README 给出的开发流程是：克隆仓库、进入具体应用目录、安装依赖并启动开发服务器。因此它是**同仓多应用**项目；当前文档没有说明用一条命令同时启动全部应用。

## 与机器人研究/工程的关联点

- **工程软件入口：** 仓库列出 CAD、Parametric、Circuits 与 Building Design 等应用，可作为开源工程软件生态的观察入口。
- **边界：** README 未将这些应用定位为机器人专用 CAD、仿真器或训练环境；不要据此推断其支持机器人模型、动力学仿真或 ROS 工作流。
- **收录范围：** 当前仅归档原始项目来源，不升格为机器人 wiki 实体页。

## 参考链接

- 官方仓库与 README：<https://github.com/HeyPuter/everything>
- 仓库许可证：<https://github.com/HeyPuter/everything/blob/main/LICENSE>
