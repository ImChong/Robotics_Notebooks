# Issue #3490 · Batch 01（已编辑节点与下一批排序）

> 核验对象：`main` 的两条路线及其知识地图 Stage 直接引用，统计口径和步骤见 [Issue #3490](https://github.com/ImChong/Robotics_Notebooks/issues/3490)。本批**不宣称**全部关联节点达到 3/3。

## 本批范围

- `roadmap/depth-rocket-landing.md`：Stage 0–6，共 **38 个唯一 wiki 节点**。
- `roadmap/depth-pcb-design.md`：Stage 0–7，共 **36 个唯一 wiki 节点**。
- 两条路线之间无重复 Stage 节点，合计 **74 个唯一节点**（只统计 `## Stage` 内的站内 wiki 链接；不把结尾的相关页面汇总误计为 Stage）。
- 两条路线页均有首屏导读、≥5 个站内链接及 Stage 标题，结构健康度各 **3/3**。

## 本批实际变更（核验现有归档，不伪造新的通读记录）

| 节点（关联 Stage） | 变更前 → 后 | 依据与实际修复 |
|---|---|---|
| `wiki/concepts/powered-descent-guidance.md`（火箭 S0） | **2/3 → 3/3** | 关联既有已标记通读的 Blackmore 2010 原文归档；原有阅读骨架齐全 |
| `wiki/methods/lossless-convexification.md`（火箭 S2） | **2/3 → 3/3** | 关联 Blackmore 2010 原文归档；其已有全文 §I–VI、附录阅读记录 |
| `wiki/methods/successive-convexification.md`（火箭 S3） | **2/3 → 3/3** | 关联 Szmuk & Açıkmeşe 2018 的现存已通读论文归档 |
| `wiki/formalizations/unit-quaternion-so3.md`（火箭 S0） | **1/3 → 3/3** | 补齐「一句话定义」，关联既有 Diebel 2006 通读归档，修正 scalar-first Hamilton 积向量部分叉乘项符号（应为 `+`） |
| `wiki/entities/kicad.md`（PCB S1、S7） | **1/3 → 2/3** | 增加定义、明确原理图到 PCB 的核心流程骨架；未标记原文通读 |
| `wiki/entities/tinkercad.md`（PCB S0） | **1/3 → 2/3** | 补齐阅读骨架「一句话定义」；未标记原文通读 |
| `wiki/entities/wokwi.md`（PCB S0、S4） | **1/3 → 2/3** | 补齐定义、核心原理、固件与外设工程验证步骤及局限；未标记原文通读 |

评分依据为 `scripts/generate_link_graph.py` 的三项规则和 `scripts/original_reading.py` 中归档状态判据。上述为按同一规则逐项推得的本批文件分数，**并非本地运行 `report_health.py` 的输出**；运行验证仍待 CI / 完整仓库环境复核。

### 已存在的一手归档通读证据

- `sources/papers/blackmore_2010_minimum_landing_error_pdg.md`：`通读：已完成`；记录全文 §I–VI 与附录核对，公式字符抽取有限制（已记入档案）。
- `sources/papers/szmuk_2018_scvx_6dof_free_final_time.md`：`通读：已完成`；记录全文 §I–V、表格和参考文献核对，图片文字层有限制。
- `sources/papers/diebel_2006_representing_attitude_quaternions.md`：`通读：已完成`；记录旋转矩阵 / 欧拉 / 四元数 / 旋转向量正文重点核对范围，12 轴序逐式验证尚未完成。

这里复用的是仓库**已有**的通读记录，未声称本批重新完整逐篇读完，也未将没有完成记录的 KiCad / Wokwi / Tinkercad 标记为已读。

## 全量可复核盘点工具

新增 `scripts/audit_depth_health.py`：

```bash
# 全 30 条路线，输出每条 Stage→节点、去重后的 0/1/2/3 分与缺项
python3 scripts/audit_depth_health.py --write

# 按路线图中历史起点的先后顺序逆序，先看最新纵深
python3 scripts/audit_depth_health.py --latest 1

# 本批被更改的 wiki 页分值复查
python3 scripts/report_health.py --changed

# 仓库现有 CI 预检
make ci-preflight
```

`--write` 应产出 `docs/quality/depth-health-backlog.md` 和 `docs/quality/depth-health-backlog.csv`。**截至本批提交，尚未在完整本地仓库执行上述命令，因此不报告未经核实的全量唯一节点数、总体已完成数或 CI 全绿。** 后续应运行后将输出作为增量进度基线。

## 后续排序（用户确认）

纵深路线排序按 `roadmap/README.md` 的「方向起点里程碑年份」**从新到旧**，而不是按文件最后修改时间或者创建日期。最新主题优先：

1. **世界–动作模型（WAM）**：`roadmap/depth-wam.md`，起点 2026。
2. **具身数据**：`roadmap/depth-embodied-data.md`，起点 2023。
3. **Real2Sim**：`roadmap/depth-real2sim.md`，起点 2023。
4. **VLA**：`roadmap/depth-vla.md`，起点 2023。
5. **Diffusion / Flow Matching**：`roadmap/depth-robotics-diffusion-dit-flow.md`，起点 2023。

同年份顺序以 README 自底向上为准；每批只填真实缺项，未能通读的保留阻塞，**#3490 不因本 PR 关闭**。
