# BotLab / MotionCanvas（地瓜机器人）

- **类型**：网站 / 在线工具（浏览器端）
- **入口**：<https://botlab.d-robotics.cc/>
- **主体**：[地瓜机器人（D-Robotics）开发者社区](https://www.d-robotics.cc/) 生态内的 Web 应用；页面 `<title>` 为 **MotionCanvas**（与港中文 Adobe MotionCanvas 论文项目同名，此处为独立产品）。
- **收录日期**：2026-05-12
- **抓取说明**：首页为 Vite SPA，公开能力以 **2026-05-12** 拉取的 `index-*.js`、`useLocaleStore-*.js` 前端资源中的 UI 文案与节点注册表为准（不含需登录的后台能力）。

## 一句话

在浏览器里用 **节点图** 把 **观测 → 处理 → ONNX 推理 → 控制输出 → MuJoCo 仿真** 串起来，并支持 **MSCP** 图导入导出、**Netron** 模型结构预览与 **Unitree G1 / Go2** 相关模板。

## 为什么值得保留

- 把 Isaac 系 RL 里常见的 **obs 构造 / history stack / policy ONNX** 搬到端侧可视化编排，降低「改一行 obs 就要重编译」的摩擦。
- 明确区分 **Strict**（推理完成再步进）与 **Pipelined / Fast**（仿真持续跑、用最新推理动作）两种 **控制同步模式**，对应真机/仿真里常见的同步语义讨论。
- **History Buffer** 同时提供 **IsaacGym（整条 obs 按帧堆叠）** 与 **IsaacLab（单 obs 先堆叠）** 两种堆叠语义选项，便于与训练侧张量布局对齐。

## 公开功能要点（来自前端资源）

| 模块 | 代表节点 / 能力 |
|------|------------------|
| Observation | Base height / linear & angular velocity / projected gravity / commands / joint pos & vel / previous actions / depth image / obs warehouse；G1 控制命令、IMU 等 |
| Processing | History buffer、math、clamp、reorder、PD、mux/demux、signal send/receive |
| Inference | ONNX Runtime（**WASM / WebGPU**，不可用时回退 WASM）；**Netron Preview** |
| Control | Action 输出、**MuJoCo** 节点（画布上限制 **单个** MuJoCo 节点） |
| Visualization | Oscilloscope、Data logger |
| 场景 / 地形 | 机器人、地形（楼梯、悬空台阶、斜坡、方块障碍、粗糙地面等）、地面颜色 |
| 图数据 | **Export / Import MSCP**（依赖 X6 图初始化） |
| 模板 | 侧边栏 **Templates**；含 **G1 WBC**、**Go2 MuJoCo Policy** 等节点族；内置多份示例 **`.onnx`**（如 `dance12_0207_1.onnx`、`loco_0731.onnx`、`amp_0309_1.onnx`、`Unitree-G1-AMP-Flat_model_30000.onnx`、`gym_policy.onnx`） |

## 对 wiki 的映射

- 升格页面：[wiki/entities/botlab-motioncanvas.md](../../wiki/entities/botlab-motioncanvas.md)

## 参考链接

- BotLab 站点：<https://botlab.d-robotics.cc/>
- 地瓜机器人社区首页：<https://www.d-robotics.cc/>
