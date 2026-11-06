# soccerbot-humanoid.github.io（SoccerBot 项目页仓库）

> 来源归档

- **类型：** repo（GitHub Pages 站点，非训练代码）
- **链接：** <https://github.com/soccerbot-humanoid/soccerbot-humanoid.github.io>
- **核查：** 2026-11-06 `git ls-remote` 可达（main = `7cb1106`），`git clone --depth 1` 只读检出；未运行任何代码
- **提交：** `7cb1106` Restore CoRL BibTeX citation with arXiv link（2026-11-05）
- **许可：** 仓库根目录**无 LICENSE / README**。`static/vendor/NOTICE.txt` 仅列第三方：MuJoCo 3.17.0（Apache-2.0）、ONNX Runtime Web 1.22.0（MIT）、Three.js 0.180.0（MIT），并声明这些许可"不为项目的机器人资产或学习权重授予新许可"。

## 目录结构

- `index.html`、`static/`（css、images、videos、vendor）
- `play/`：`index.html`、`play.js`、`play.css`、`football.js`、`simulation.js`、`simulation-worker.js`、`bundle/`（`policy.onnx` 2.75 MB、`scene.xml`、`manifest.json`、32 个 `mesh-*.stl.gz`）

## 入口与关键模块（静态阅读）

- `play/simulation-worker.js`：`onmessage` 收 `init` → `BrowserSimulation.create` → `reset(preset)`；`tick` 循环 `action` → `advance` → `postMessage(snapshot)`。
- `play/simulation.js`：`create`（加载 MuJoCo 模型并校验 nq=43、nv=41、nu=29，建 ONNX 会话）、`buildObservation`（106 维）、`updateControl`（技能选择）、`advance`（`policy.run` → 动作缩放 → 4 个物理步 PD + 延迟）。
- `bundle/manifest.json`：策略名 "Exp7 · FailRecover"，检查点名含 `LocalMove106D_PlanarKick_RollingTTA_FailRecover_..._noPrior_noLAB_...`；`policy_dt=0.02`、`physics_dt=0.005`、`decimation=4`、`delay_max=2`；源 MuJoCo 3.4.0、浏览器 3.17.0；记录 policy / onnx / scene 的 SHA-256。
- 技能：代码里 `skills=['dribble','kick','move','trap']`，但 UI 只触发 dribble、kick 与站立恢复（move），无停球。

## 对 wiki 的映射

- [paper-soccerbot-latent-motion-priors](../../wiki/entities/paper-soccerbot-latent-motion-priors.md)：源码运行时序图一节
