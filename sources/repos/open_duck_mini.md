# open_duck_mini

> 来源归档

- **标题：** Open Duck Mini
- **类型：** repo
- **来源：** apirrone（Antoine Pirrone，HuggingFace / Pollen Robotics 赞助）
- **链接：** https://github.com/apirrone/Open_Duck_Mini
- **Stars：** ~3.0k（2026-05，GitHub）
- **入库日期：** 2026-05-28
- **一句话说明：** 迪士尼 BDX 双足娱乐机器人的开源迷你复刻：Onshape CAD、BOM（v2 目标 &lt;$400）、Feetech 舵机机体，并作为 Open Duck 多仓生态的总入口（仿真训练、参考运动、机载 Runtime 分仓维护）。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-duck-mini.md`](../../wiki/entities/open-duck-mini.md)

---

## 项目定位

- **灵感来源：** Disney Imagineering 的 [BDX Droid](https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/)（BD-X 小型双足角色机器人）。
- **形态：** 双足、紧凑、娱乐/教育向 DIY 平台；**非**研究级全尺寸人形，但完整覆盖「CAD → 仿真模型 → RL → 真机」闭环。
- **社区：** Discord https://discord.gg/UtJZsgfQGe；Tnkr 中文/英文组装指南；HuggingFace、Pollen Robotics 赞助。

## 版本线

| 版本 | 身高（腿伸展） | BOM 目标 | 训练栈 | 说明 |
|------|----------------|----------|--------|------|
| **v1 / alpha** | ~35 cm | 较高 | Isaac Gym + [AWD](https://github.com/rimim/AWD) | 早期原型；机械间隙大、难组装；README 建议等待 v2 |
| **v2（当前主线）** | ~42 cm | **&lt;$400** | [Open_Duck_Playground](open_duck_playground.md)（MuJoCo Playground / MJX） | 机械设计基本定稿；sim2real 行走已公开演示；预训练 ONNX 策略可下载 |

**默认分支：** 开发集中在 `v2` 分支；主 README 与 CAD/BOM 以 v2 为准。

## 仓库职责（Hub）

本仓集中存放与整机相关的资源，并链接到专用子仓：

| 子系统 | 仓库 |
|--------|------|
| RL 训练（MuJoCo Playground） | [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) → [open_duck_playground.md](open_duck_playground.md) |
| 参考运动 / 模仿奖励 | [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) → [open_duck_reference_motion_generator.md](open_duck_reference_motion_generator.md) |
| 机载部署（Raspberry Pi Zero 2W） | [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) → [open_duck_mini_runtime.md](open_duck_mini_runtime.md) |

## 硬件与制造

- **CAD（v2）：** [Onshape 公开文档](https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05)
- **BOM：** [Google Sheets（v2）](https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing)；中文采购见飞书 wiki（README 链接）
- **执行器：** 腿部由 `xc330-M288-T`（较 v1 的 `xl330` 更强）等 Feetech 总线舵机驱动；v1 曾用 xl330
- **导出仿真：** `onshape-to-robot` → URDF/MJCF；Playground 仓内 `xmls/config.json` 做 MJX 轻量化
- **执行器辨识：** [Rhoban BAM](https://github.com/Rhoban/bam) → STS3215 等参数导出至 MuJoCo（`damping` / `kp` / `frictionloss` / `armature` / `forcerange`）

## 文档与 Sim2Real 要点（v2）

- **组装：** [Tnkr v2 项目文档（主线）](../sites/tnkr-open-duck-mini-v2.md)（https://tnkr.ai/open-duck-mini/open-duck-mini-v2）、`docs/assembly_guide.md`（进行中，与 Tnkr 步骤对齐）、`docs/print_guide.md`
- **Sim2Real 管线说明：** `docs/sim2real.md`（准确 MJCF + BAM 电机模型 → Playground 训练 → Runtime 上机）
- **预训练策略：** `BEST_WALK_ONNX.onnx` / `BEST_WALK_ONNX_2.onnx`；MuJoCo 回放 `v2_rl_walk_mujoco.py`

## 相关论文与资源（项目 README 收录）

- Disney BDX 设计与控制：[Disney Research 论文页](https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/)、[GTC 2024 讲座](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63374/)
- 模仿奖励：BDX 论文中的 imitation reward；参考运动由 Placo 参数化步态生成
- Sim2Real 博客：https://www.haonanyu.blog/post/sim2real/
- 其它：DRLoco、DeepMimic、MuJoCo MPC 等链接见上游 README

## 与本仓库 wiki 的映射

| 主题 | 建议 wiki |
|------|-----------|
| 整机与生态 | `wiki/entities/open-duck-mini.md` |
| MuJoCo Playground 训练 | `wiki/entities/open-duck-playground.md` |
| 参考运动生成 | `wiki/entities/open-duck-reference-motion-generator.md` |
| 机载 Runtime | `wiki/entities/open-duck-mini-runtime.md` |
| 娱乐双足 / BDX 路线 | `wiki/methods/disney-olaf-character-robot.md`（机构与奖励哲学对照） |
| Sim2Real | `wiki/concepts/sim2real.md` |
