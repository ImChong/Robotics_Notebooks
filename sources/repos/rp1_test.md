# rp1_test（RP1 模块老化与整机可靠性测试平台）

> 来源归档

- **标题：** RP1 Test — Module aging and whole-robot reliability testing for RP1
- **类型：** repo（测试台架硬件 + 可靠性软件平台 + 测试方法）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/rp1_test
- **入库日期：** 2026-10-27（核查快照：`main` @ `3eacb00`，2026-10-19 PR #1 `codex/docs-actionable-guides-20261019`；`VERSION` = 0.2.0）
- **一句话说明：** 手臂 / 胸腔 / 腿部负载三个台架的机械资料、控制柜参考架构，配 React + FastAPI + PostgreSQL 可靠性平台、PySide6 PLC/CAN 上位机，及 18 项整机可靠性测试方法草案。
- **沉淀到 wiki：** 是（[`wiki/entities/rp1-test.md`](../../wiki/entities/rp1-test.md)）
- **上级目录：** [RP1 总览仓库归档](roboparty_rp1.md)

## 开源状态

- **已开源（源码 + 台架设计 + 测试方法）**，约 674 个受控文件：`software/{frontend,service,database,host}`、`hardware/`、`test-cases/`。
- **分材料许可**：`software/` 与脚本 Apache-2.0；`hardware/` CERN-OHL-W-2.0；README / 测试方法 CC-BY-SA-4.0；原硬件文档保留 MIT 选项；`software/host/motors/` 保留 GPL-3.0；供应商 CAD、第三方 SDK 保留各自权利（`LICENSES.md`、`THIRD_PARTY.md`）。真实测量 / 生产数据需另行数据许可。
- **未提供**：电气施工原理图、I/O 表、端子表、逐芯线束图与 PLC 工程（`hardware/docs/integration.md` 明说）；三个台架 Parasolid 总装模型以 Release `v0.1.0` 的 `RP1_Test_Parasolid_CAD.zip` 分发，本次未下载。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `test-cases/whole-body/SYS-REL-001…021-*.md` | 18 项整机可靠性用例（编号不连续，ID 稳定） |
| `test-cases/{common,checklist,open-items,_template}.md` | 公共规则、执行记录、待确认参数、模板 |
| `software/compose.yaml`、`software/VERSION`（平台 2.0.6） | Docker Compose 部署（postgres:16 等） |
| `software/host/` | PySide6 HMI：`factory_hmi/{desktop,gateway,plc,core,sync}`、Modbus TCP 协议 V1.0/1.1/1.2 文档 |
| `software/database/alembic`、`software/service/app`、`software/frontend/src` | 迁移、业务服务、React 大屏 |
| `hardware/mechanical/{手臂胸腔,腿部}`、`hardware/electrical/system-bom.csv` | 三台架资料、LFD835 V1.9 四通道控制柜采购表 |
| `docs/architecture.md`、`software/docs/data-workflow.md` | 架构与数据归档 / 审批流程 |

## 对 wiki 的映射

- [wiki/entities/rp1-test.md](../../wiki/entities/rp1-test.md)
- 关联：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[romomo](../../wiki/entities/romomo.md)、[humanoid-testing-workflow](../../wiki/concepts/humanoid-testing-workflow.md)
