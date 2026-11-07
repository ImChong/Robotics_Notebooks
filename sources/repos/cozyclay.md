# CozyClay — 官方源码与技术文档归档

- **标题：** CozyClay — Open Source Browser Previsualization Studio
- **类型：** repo
- **维护者：** Doyun（HaD0Yun）／NomaDamas
- **原始用户链接：** <https://github.com/HaD0Yun/CozyClay>
- **官方 README 所用仓库：** <https://github.com/NomaDamas/CozyClay>
- **项目官网：** <https://cozyclay.org/>
- **入库日期：** 2026-11-07
- **核对版本：** main，`6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9`（`package.json` 3.2.2；2026-11-07）
- **核对范围：** 官方 README、`INSTALL_FOR_AGENTS.md`、`docs/motion3-agent-guide.md`、`mcp/README.md` 及 `package.json`；源码重点阅读 `bin/cozyclay.mjs`、`mcp/server.mjs`、`mcp/tool-handlers.mjs`、`src/command-bus.js`、`src/commands/motion3-compile.js`、`src/motion3/{compile-worker.js,compile-job.js,pipeline.js,compiler/compile.js,qa/index.js}`、`src/shot.js`。范围为官方关键文档和执行链的源码核对，**不等同于对整个仓库逐文件审计或运行测试**。
- **通读：** 已完成
- **一句话说明：** 用浏览器搭场景、编角色动作与摄影机镜头，将经过编译和 QA 的灰盒预演、关键帧及提示词交给外部 AI 视频模型。
- **对 wiki 的映射：** [CozyClay 主实体](../../wiki/entities/cozyclay.md)

## 定位与开源核对

| 项目 | 一手资料结论 |
|---|---|
| **开放程度** | 已开源：包含浏览器工作室、MCP 工具、编译器、QA 与命令行入口 |
| **许可** | `AGPL-3.0-or-later`；具体分发及网络服务合规应读 `LICENSE` / `LICENSING.md` |
| **技术栈** | JavaScript、React 19、Three.js、React Three Fiber、Vite；Node.js 服务端与 MCP |
| **运行环境** | Node.js ≥22.19、Chromium 浏览器；基础预演不要求 GPU、Python 或模型下载 |
| **主要模式** | Animation：rig 角色运动；Storyboard：分镜板、animatic |
| **AI 接口** | 工作室 Agent 面板、MCP、`cclay live` 命令行；视频模型为导出成果的下游 |
| **运行版本** | `3.2.2`（固定 SHA 的 `package.json`）；后续版本功能可能变化 |

## 官方文档提炼

1. **Stage：** 场景基元、rig 角色或胶囊占位角色、道具布局。
2. **Motion：** Agent 写 motion spec（关键姿态、锚点、技能、行走、物体与约束）；`motion.compile` 将 spec 确定性编译为逐帧 rig 动作并运行 QA。返回 `qa.groups` 的 cause/hint 后可调整 spec 再编译。
3. **Camera：** 镜头、焦距、机位、轨道、镜头切换；`src/shot.js` 将三维相机几何转换为镜头语言和 prompt。
4. **Export：** 首末关键帧、灰盒参考片段、镜头信息、提示词；另有 MP4、depth/normal、contact sheet、OTIO。
5. **可选生成：** `motion.fill` 在自由关键姿态间使用 MotionBricks 补间，**非**默认 `motion.compile` 必选步骤；应区分本地模型补间与确定性编译。
6. **Agent 接口：** `npx -y cozyclay mcp` 提供 MCP；`cclay live status --wait` 连接已打开的 Studio；未连接浏览器时仅部分工具支持无头内存场景。

## 源码路径核对

| 入口/模块 | 可验证的责任 |
|---|---|
| `bin/cozyclay.mjs` | `npx cozyclay` 的本地 HTTP studio 启动器、静态资源与路由 |
| `mcp/server.mjs` | MCP stdio／Streamable HTTP、工具注册和 Live Hub |
| `mcp/tool-handlers.mjs` | 场景/相机/项目工具与 Live 工作区路由 |
| `src/command-bus.js` | Studio action 参数校验、事务/撤销与回执 |
| `src/commands/motion3-compile.js` | `createCompileRunner` 负责 Worker 池和无 Worker 回退 |
| `src/motion3/compile-worker.js` | Worker `answerOf` 调用 |
| `src/motion3/compile-job.js` | `runCompileJob` 封装 spec、搜索/修正、QA、bake take |
| `src/motion3/pipeline.js` | `compileSpec`：扩展、校验、编译、QA、观察值 |
| `src/motion3/compiler/compile.js` | `compileMotion`：场景+spec 的确定性逐帧求解 |
| `src/motion3/qa/index.js` | `runMotionQa` 汇总碰撞、脚滑、接触、reach、负载等问题 |
| `src/shot.js` | `deriveShot` / `composePrompt`：构图数据与自然语言镜头提示 |

> 代码时序只基于上述实际入口；不将 `motion.compile` 误记成扩散式运动生成或物理机器人控制闭环。

## 快速体验（官方 README）

```bash
npx cozyclay
# Chrome / Edge 等 Chromium 浏览器访问 http://127.0.0.1:5180/app/
# MCP: npx -y cozyclay mcp
```

源码开发：`git clone https://github.com/NomaDamas/CozyClay.git` → `npm install` → `npm run dev`。官方 `INSTALL_FOR_AGENTS.md` 给出 `--no-star`、`--no-open` 和 live/MCP 连接步骤。

## 与机器人研究的边界

CozyClay 输出 **视觉预演、角色运动和镜头资产**，可用于人机交互演示、数据故事板、机器人展示视频的分镜规划；其 QA 使用场景几何代理检查动画，不等于 MuJoCo/Isaac Lab 动力学、稳定控制器或可上真机的轨迹验证。

## 一手资料

- [README（固定版本）](https://github.com/NomaDamas/CozyClay/blob/6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9/README.md)
- [INSTALL_FOR_AGENTS（固定版本）](https://github.com/NomaDamas/CozyClay/blob/6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9/INSTALL_FOR_AGENTS.md)
- [Motion 3 Agent Guide](https://github.com/NomaDamas/CozyClay/blob/6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9/docs/motion3-agent-guide.md)
- [MCP README](https://github.com/NomaDamas/CozyClay/blob/6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9/mcp/README.md)
- [动作 Worker 入口](https://github.com/NomaDamas/CozyClay/blob/6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9/src/commands/motion3-compile.js)
- [许可证](https://github.com/NomaDamas/CozyClay/blob/6635fd56f4fa0c93c5ccf9a18cc85bf42823b2b9/LICENSING.md)

## 相关归档

- [官网归档](../sites/cozyclay-org.md)
- [Wiki 主节点](../../wiki/entities/cozyclay.md)
