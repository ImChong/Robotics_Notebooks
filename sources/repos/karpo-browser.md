# karpo-browser：System 1 / System 2 浏览器代理与源码归档

- **标题：** karpo-browser — A browser for AI agents
- **类型：** repo / TypeScript 浏览器执行工具、MCP Server、CLI 与 SDK
- **开发方：** MachinePulse Pte. Ltd.
- **代码：** https://github.com/machinepulse-ai/karpo-browser
- **专用权重：** https://huggingface.co/MachinePulseAI/karpo-decision-27B-act
- **相关评测：** https://github.com/machinepulse-ai/karpo-browserbench
- **核查版本：** 上游 main（2026-11-21，所读 GitHub 树 9319835e）
- **核对范围：** 通读 README 的机制、CLI/MCP/SDK、运行配置、Secret 安全、Limitations 与许可证；阅读 `src/index.ts` 的 Browser API、`src/run/run.ts` 的 Run 状态机与输入/结果、`src/run/act.ts` 的真实 CDP 动作/页面 settle、`src/mcp.ts` 的工具注册、`AGENTS.md` 与 `package.json` 的源码目录/运行入口；未运行 Chrome 或在线基准。
- **通读：** 已完成
- **许可：** Apache-2.0（源码）。
- **开放状态：** 代码仓含可执行 Node 包、CLI、MCP、CDP 层、TypeScript SDK、Vitest 测试与可选 System 2 控制器；官方专用权重单独托管。
- **沉淀到 wiki：** [karpo-browser](../../wiki/entities/karpo-browser.md)

## 源码事实

- `src/index.ts` 中 `Browser.launch` / `Browser.connect` 管 Chrome / 既有 CDP WebSocket，`Browser.open` → `Browser.run` 进入 `src/run/run.ts` 的 `Run.result`。
- `src/run/run.ts` 重复读取当前页和截图，`src/decider/request.ts` 整理目标、值、页面候选和循环判断问题，`src/decider/systemone.ts` 发 `state + questions`。
- 结果为一条决定（例如 `CLICK 5`）或终止原因；`src/run/act.ts` 的 `Acting` 检查元素仍在原位、角色/名称不变，再以 CDP 发真实点击、输入、滚动，等待页面 settle 并把结果放进历史。
- `src/mcp.ts` 暴露 `open`、`run_browser_agent`、`read_page`、`screenshot` 与手动动作；`test/controller/` 提供可选通用语言模型 System 2 控制器，不是运行每个点击所必需的组件。
- `secrets` 不放入 System 1 请求并在后续页面读取时屏蔽；但模型决策的 `NEEDS_APPROVAL` 不是底层安全硬阻断，调用方须独立审批。

## 官方运行入口

```bash
npm install -g karpo-browser
export SYSTEM1_MODEL=typesafe/jev-1.14
export SYSTEM1_API_KEY=...
karpo-browser run --url https://en.wikipedia.org/ --goal "Open the Wikipedia article about the Apache License" --value "search=Apache License"
# 任意 MCP Client：
npx -y karpo-browser mcp
```

要求 Node.js ≥22.18 与 Chrome/Chromium 或既有 CDP 地址。源码库 `npm run check`、`npm test` 和 `npm run build` 提供静态检查/测试/编译。

## 限制与交叉映射

- System 1 只负责页面动作，System 2 才能读长列表、比较、写结论；无法自动保证真实世界任务安全。
- 不可假定专用 27B Act 推理服务直接兼容仓库客户端，端点路径差异需单独适配；模型卡明确此边界。
- [27B Act 模型归档](../models/karpo-decision-27b-act.md)；[BrowserBench 源码归档](./karpo-browserbench.md)。

