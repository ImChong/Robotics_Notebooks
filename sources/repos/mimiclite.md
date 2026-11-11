# MimicLite

> 来源归档

- **标题：** MimicLite
- **类型：** repo（监督学习 / 运动跟踪基础设施）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/MimicLite
- **入库日期：** 2026-07-14（仓库复核 2026-10-27）
- **再次核查：** 2026-11-11
- **核对范围：** 完整通读 Roboparty/MimicLite 英文 README.md、中文 README_cn.md、.gitmodules；EGalahad/mimic-lite README.md、docs/checkpoints.md；EGalahad/sim2real README.md。结合官方《MimicLite 使用指南》第 1–6 章复核；阅读训练源码 scripts/train.py、scripts/train_sequential.py、scripts/play.py 与 scripts/eval.py 的关键执行入口和职责。不声称逐行通读所有源文件。
- **通读：** 已完成
- **阅读边界：** 本条的「通读」特指上述官方 README 与配套文档、源码核验范围，**不包含未能获取全文的 mimic-lite.pdf 技术报告**；后者不得标记为已通读。
- **官方技术文档归档：** [MimicLite 使用指南全文核验](../sites/mimiclite-official-guide-2026-11-11.md)
- **一句话说明：** 面向人形机器人通用运动跟踪的开源训练与部署基础设施，贯通数据、训练、统一评测与真机部署，支持小时级迭代与跨 codebase 策略接入。
- **沉淀到 wiki：** 是（`wiki/entities/mimiclite.md`）

---

## 核心能力（文内自述）

| 能力 | 说明 |
|------|------|
| 小时级训练 | 8×RTX 4090、约 3h 训练通用跟踪策略；约 24 GPU-hours |
| Tracking Infra | any4hdmi 统一多来源动作；mjhub 保证训练/运动学/sim2sim 一致性 |
| 遥操 + 高动态 | 同一 policy 支持 Pico/XR 低延迟遥操与高动态真机动作 |
| 跨 codebase 部署 | 模块化 observation interface；已接入 SONIC、HEFT、TeleopIT、Humanoid-GPT、BFM-Zero、TWIST2 |

---

## 开源状态（2026-10-27 复核）

- **已开源（落地页 + 外部子仓库）**：`Roboparty/MimicLite` 为 GPL-3.0-or-later 项目落地页（含 `mimic-lite.pdf` 技术报告），子模块 `EGalahad/mimic-lite`、`Agent-3154/active-adaptation`、`EGalahad/sim2real`，数据工具 `EGalahad/any4hdmi`。
- 已发布 G1 / H2 / A3 / RP1(28 与 24 DoF) 的 Finetune 阶段 ROA student checkpoint（Google Drive）；README 对比图仅覆盖 G1。
- README 未出现 `mjhub` 命名（文内命名见下）。

---

## 关键子组件（文内命名）

- **any4hdmi** — 将 LAFAN、100STYLE、SONIC、真机数据等组织为统一 motion 格式
- **mjhub** — 机器人模型在训练、运动学计算与 sim2sim 验证中的一致性层

---

## 2026-11-11 源码和发布配置复核

- **训练脚本：** `projects/mimic-lite/scripts/train.py` 的 `main(cfg)` 依赖 `active_adaptation.helpers.make_env_policy` 构建任务/策略；`train_sequential.py` 按 stages 依次调用训练脚本并传递前一阶段 checkpoint。
- **导出脚本：** `scripts/play.py::export_policy` 通过共享的 `active_adaptation.utils.export.export_onnx` 生成 ONNX，同时写部署 YAML；`scripts/eval.py::fixed_step_eval` 执行定步验证。
- **G1 发布模型：** PPO（92.3 GPU-h，16×16384，1024×3）、ROA（173.2 GPU-h，16×16384，1024×3）、v1.1 ROA student（50.106 GPU-h，8×8192，512×3）。它们**不是同一权重或同一训练成本**，不能直接用主页面「8×4090 约 3h」覆盖实际发布条件。
- **真实部署：** 仿真/实机由 `sim2real` 统一 ONNX+YAML 运行时承接；使用 `--robot-io inline` 以减少机器人通信抖动；`PICO` 通过动作重定向提供实时参考。
- **已确认限制：** `docs/checkpoints.md` 报告 A3 原生检查 67/72 通过、长动作 root drift，集成 ONNX / sim2real 部署仍未验证；使用 checkpoint 时必须匹配机器人资产与训练配置。BONES-SEED 的许可证禁止随意再分发。
- **PDF 边界：** 顶层 GitHub 仓库的 `mimic-lite.pdf` 技术报告存在，但本次无法读取其全文；因此本复核不从 PDF 推断尚未在上述源文证实的内容。

## 对 wiki 的映射

- [mimiclite](../../wiki/entities/mimiclite.md)
- 父级：[party-os](../../wiki/entities/party-os.md)
- 交叉：[sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)
