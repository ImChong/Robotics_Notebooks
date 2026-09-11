# NVlabs/SoL-Pi

> 来源归档

- **标题：** SoL-Pi — Scaling Auto-Research Loops for Efficient Agent Harnesses
- **类型：** repo
- **来源：** NVIDIA NVLabs
- **链接：** <https://github.com/NVlabs/SoL-Pi>
- **项目页：** <https://nvlabs.github.io/SoL-Pi/>
- **许可：** MIT
- **入库日期：** 2026-09-11
- **一句话说明：** Pi（`@earendil-works/pi-coding-agent`）上的独立效率扩展包：四条 opt-in 机制（Action Fusion、ObservationPack、Evidence-Preserving Reducer、Online Context Compact），不改 Pi 源码、默认全关。
- **沉淀到 wiki：** 是 → [`wiki/entities/sol-pi.md`](../../wiki/entities/sol-pi.md)

## 开源状态（步骤 2.5）

- **代码：** 完整 TypeScript 扩展于 `src/sol-pi/extensions/*`；`docs/configuration.md`、`docs/compatibility.md`。
- **安装：** `npm install -g @earendil-works/pi-coding-agent@0.84.2` → `pi install git:github.com/NVlabs/SoL-Pi`（可 `--local --approve`）。
- **结论：** **已开源**；README 声明 **非 Pi 官方发行版**，仅 public Pi extension API。

## README 要点（归纳）

| 机制 | 目录 | 行为摘要 |
|------|------|----------|
| Action Fusion | `extensions/action-fusion/` | 编辑/写入后本地执行 follow-up 命令，单次 observation |
| ObservationPack | `extensions/observation-pack/` | 大文本 → handle + 分页 recall |
| Evidence-Preserving Reducer | `extensions/evidence-preserving-reducer/` | 归档 log + 核验 receipt 后再给 frontier |
| Online Context Compact | `extensions/online-context-compact/` | 计划步完成时触发 Pi 原生 compaction 候选 |

**设计约束：** 无 Pi patch；缺配置则机制关闭；证据本地保留；认证/模型/shell 仍由 Pi 控制。

**依赖：** Node.js ≥ 22.19；Pi **0.84.2**（README 测试版本）。

## 对 wiki 的映射

- 升格实体页：[SoL-Pi](../../wiki/entities/sol-pi.md)
- 对照 [DeepSeek Harness](../../wiki/entities/deepseek-harness.md) / [OpenClaw](../../wiki/entities/openclaw.md) — 通用 coding harness 选型轴
- 对照 [HarnessBank](../../wiki/entities/paper-harnessbank.md) — 冻结模型下的 harness 搜索与门控

## 参考来源（原始）

- GitHub：<https://github.com/NVlabs/SoL-Pi>（2026-09-11）
- 项目页：<https://nvlabs.github.io/SoL-Pi/>
