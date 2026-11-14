# Issue #3490 · Batch 02：WAM 最新纵深优先

> 优先级按 `roadmap/README.md` 的**研究方向历史起点**倒序，不按文件创建/修改日期。本批从最新主题 `roadmap/depth-wam.md` 开始。

> **后续验收（2026-11-14）：** 本页保留 Batch 02 的历史记录；WAM 的 62 个知识节点现已全部核验为 3/3，完整前后分数、Stage 与原文证据见[本轮验收报告](wam-depth-health.md)。

## 范围与基线

- **路线**：`roadmap/depth-wam.md`，**Stage 0–5**。
- **关联节点**：Stage 中出现的 wiki 站内链接去重后共 **62 个**。
- **路线页健康度**：首屏导读 + ≥5 条内部链接 + Stage 结构，按 `compute_roadmap_health` 判据 **3/3**。
- 本批实质修改 **10 个 wiki 节点**；它们在原有来源/摘要条件下均属于「缺阅读骨架、没有完成原文归档」的分数段。
- 说明：上述 Stage 节点数来源于当前 `main` 路线页结构检查，不代表所有 62 个节点均已逐篇审校。本批未跑全量本地 `report_health.py`；后续须复核。

## 本批节点（按 Stage）

| Stage | 节点 | 评分变化（按评分规则推定） | 补齐的内容 |
|---|---|---|---|
| 1 | `wiki/entities/tau0-world-model.md` | 1/3 → **2/3** | WAM 共享预测核心原理、部署接口与局限 |
| 1 | `wiki/entities/paper-physisforcing.md` | 1/3 → **2/3** | 两层物理监督的核验步骤、消融和失效案例 |
| 2 | `wiki/methods/defi-decoupled-dynamics-vla.md` | 1/3 → **2/3** | GFDM/GIDM 独立验证、监督分流、控制时延检查 |
| 2 | `wiki/entities/paper-being-m07-humanoid-latent-wam.md` | 1/3 → **2/3** | 三阶段训练梯度边界、22D/66D 与上层/低层频率 |
| 3 | `wiki/entities/paper-dit4dit-video-action-model.md` | 1/3 → **2/3** | 双 DiT 动作独立推理 vs 同时预测视频的验证维度 |
| 3 | `wiki/entities/paper-abot-m05-mobile-manipulation-wam.md` | 1/3 → **2/3** | 移动操作轨迹时间对齐及仿真评估边界 |
| 3 | `wiki/entities/dexmal-dw05.md` | 1/3 → **2/3** | 视频/动作/价值专家架构、部署与已知不完整发布 |
| 3 | `wiki/entities/lumo-2.md` | 1/3 → **2/3** | 将已有潜世界表示与动作生成正文纳入统一核心原理章节 |
| 3 | `wiki/methods/pelican-unified-1.md` | 1/3 → **2/3** | 联合扩散推理预算、复现组件和模型发布边界 |
| 5 | `wiki/entities/paper-gigaworld-1-policy-evaluation.md` | 1/3 → **2/3** | 评估器真机成败与世界模型动作忠实对照协议 |

**未提升至 3/3 的原因：** 以上十页所声明的 `sources/` 存档未提供满足 `original_fully_read` 判据的已完成原文通读记录；不为得分擅自添加 `original:`、伪写 `通读：已完成` 或将官方 README 当作整篇论文。具体源码实测、真机复现均未进行。

## 本路线已存在的合格通读锚点（未在本批虚增）

- `wiki/concepts/world-action-models.md` → `sources/papers/wam_survey_reading_arxiv_2605_12090.md`：已有通读记录及完整阅读骨架。
- `wiki/entities/paper-rcl-wam-robot-learning-control-survey.md` → `sources/papers/rcl_wam_survey_reading_arxiv_2609_16074.md`：已存在综述通读归档。
- `wiki/methods/generative-world-models.md` → GAIA-1、UniSim 已读归档。
- `wiki/methods/model-based-rl.md` → MBPO 已读归档。
- `wiki/overview/robot-world-models-training-loop-taxonomy.md` → 机器人世界模型综述已读归档。
- `wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md` → MotionWAM 已读归档。

## 合入前复核和后续进度

```bash
python3 scripts/report_health.py --changed
python3 scripts/audit_depth_health.py --latest 1
make ci-preflight
```

`scripts/audit_depth_health.py` 在 [PR #3504](https://github.com/ImChong/Robotics_Notebooks/pull/3504) 中，尚未合入时本批的上述命令需先在含该脚本的本地工作树运行或待其合并；**本批未声称已执行 CI**。

下一批仍按 WAM 继续逐篇补真实一手原文证据、源码时序图与评测边界；WAM 逐节点达标后，按顺序转至具身数据 → Real2Sim → VLA → 扩散/流匹配。Issue #3490 仍保持开放。
