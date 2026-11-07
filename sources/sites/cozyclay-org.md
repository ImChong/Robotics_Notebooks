# CozyClay — 官方项目主页归档

- **标题：** CozyClay — Open Source 3D Previs Software in Your Browser
- **类型：** site
- **链接：** <https://cozyclay.org/>
- **代码：** <https://github.com/NomaDamas/CozyClay>
- **维护者：** Doyun／NomaDamas
- **入库日期：** 2026-11-07
- **核对范围：** 官方项目首页的产品定位、操作方式、安装命令、控制列表及官方源码入口，并对照官方 README / 安装说明。
- **通读：** 已完成
- **一句话说明：** 浏览器内搭建 3D 场景、角色姿态与镜头，导出供 AI 视频模型使用的预演参考，减少反复试错。
- **沉淀到 wiki：** [CozyClay](../../wiki/entities/cozyclay.md)

## 一手核对

- **定位：** 开源浏览器 3D 预演 Studio；不是一套独立视频生成基础模型。
- **输入：** 场景、角色与姿态、时间线、镜头与运动指令。
- **输出：** 分镜或灰盒参考片段、关键帧/截图、相机设置、生成提示词及其他导出格式（具体格式清单核对 README）。
- **部署：** 官方展示 `npx cozyclay --via site`；源码路线见 `git clone`、`npm install`、`npm run dev`；Chromium 和 Node.js ≥22.19。
- **代码开放状态：** 已开源，仓库含前端、Motion3 编译链、MCP 与 CLI，许可 `AGPL-3.0-or-later`（以仓库 `LICENSE` 为准）。
- **容易误读：** 3D 预演的碰撞与运动学检查，不等于高保真机器人动力学仿真与 Sim2Real 验证。

## 关联

- [GitHub 源码/技术文档归档](../repos/cozyclay.md)
- [Wiki 知识页](../../wiki/entities/cozyclay.md)
- [官方 Agent 安装说明](https://github.com/NomaDamas/CozyClay/blob/main/INSTALL_FOR_AGENTS.md)
