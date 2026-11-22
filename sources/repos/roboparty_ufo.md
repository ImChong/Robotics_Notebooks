# UFO（Roboparty 无监督 RL 控制框架）

> 来源归档

- **标题：** UFO — Unsupervised RL Control Development Framework
- **类型：** repo（无监督强化学习控制框架）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/UFO
- **入库日期：** 2026-07-14（仓库复核 2026-10-27；2026-11-21 再次复核）
- **核对范围：** 完整通读官方 `Roboparty/UFO` `main` 的 `README.md`（153 行，涵盖 News/测量口径、FB/TeCH 统一 CLI、LaFAN-40 数据、8×H20 192M 步 4h45m 实测、安装、数据准备、训练、三种推理、文档入口和 CC BY-NC 4.0 许可边界）；未将独立 PDF、训练实测或 docs/ 全部文件当作本次已通读。
- **通读：** 已完成
- **一句话说明：** 面向研发、完全开源的人形机器人无监督强化学习控制开发框架，覆盖训练基础设施、数据管线、算法研究与推理部署全流程。
- **沉淀到 wiki：** 是（`wiki/entities/roboparty-ufo.md`）

---

## 核心能力（文内自述）

| 能力 | 说明 |
|------|------|
| Fast Training | MJLab backend；8×4090 <12h 完成 BFM-Zero；8×H200 6–8h |
| 通用可扩展 | 无缝适配不同机器人形态；多来源数据混合训练与调度 |
| 表征集成 | BFM-Zero（FB Representation）；[TeCH](https://lab.roboparty.com/outputs/tech)（TLDR 时间距离表征） |
| 真机遥操 | 首次开源无监督 RL 控制遥操作代码与完整验证方案 |

---

## 开源状态与现状（2026-10-27 复核）

- **已开源**，许可 CC BY-NC 4.0（非商用）；最近提交 2026-10-21，包名为 `ufo/`，旧 `humanoidverse` 布局已替换。
- 支持 G1 / RP1.3、FB / TeCH；训练与推理入口 `run_train.sh`、`run_inference.sh`；训练数据 `xuewang/UFO-MotionData`、G1 真机验证策略 `xuewang/ufo-g1-policy` 位于 Hugging Face；G1 部署见 `deploy` 分支，RP1 部署见 `Roboparty/roboparty_deploy`。
- README 声明 RP1 TeCH 策略未在真机测试；README 自报 8×H20 约 4 h 45 min 完成 192M 步 FB 训练。
- 下游：[P-BFM](../../wiki/entities/paper-p-bfm.md) 基于 UFO 构建。

---

## 命名辨析

- **UFO（本仓库）** = Roboparty 无监督 RL 控制框架，与通用缩写无关
- 与 [paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md) 的关系：UFO 集成并加速 BFM-Zero 类算法训练

---

## 对 wiki 的映射

- [roboparty-ufo](../../wiki/entities/roboparty-ufo.md)
- 父级：[party-os](../../wiki/entities/party-os.md)
- 交叉：[mjlab](../../wiki/entities/mjlab.md)、[paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md)、[paper-tech-humanoid-control](../../wiki/entities/paper-tech-humanoid-control.md)
- 成果页：[roboparty_lab_tech_humanoid_control.md](../sites/roboparty_lab_tech_humanoid_control.md)

## 2026-11-21 README 全文核对补记

- **可复现入口：** `uv sync --locked`；`scripts/download_data.sh g1_lafan|rp1_lafan` 校验 SHA256；`run_train.sh --robot g1|rp1 --agent fb|tech --gpu-ids all`；`run_inference.sh tracking|reward|goal`。其中 reward 推理只支持 FB，并要求已保存的 rank-0 replay。
- **官方测量口径：** 2026-10-15 发布的 192M-step FB 记录使用 8×NVIDIA H20、BF16、1024-wide shared-critic 网络；表中训练耗时包含初始化/评估/存档，个别时长为估算，且历史配置变动不构成严格 DR 奖励消融对照。
- **部署与许可：** README 将 G1 真机部署指向 `deploy` 分支，RP1 指向 `Roboparty/roboparty_deploy`；**RP1 TeCH 未真机测试**。主仓 CC BY-NC 4.0，仅非商业研究用途。
- **阅读边界：** 已通读上述官方 README，不宣称执行训练或核验独立 PDF 的每一页。
