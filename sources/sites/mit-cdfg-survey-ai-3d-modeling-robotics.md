# MIT CDFG · AI for 3D Modeling, Computational Design and Robotics（Living Survey）

- **类型**：site / survey-project / research-report
- **收录日期**：2026-09-29
- **全文复核日期**：2026-11-28
- **链接**：<https://mit-cdfg.github.io/Survey-AI-for-3D-modeling-Robotics/>
- **机构**：MIT CSAIL · Computational Design and Fabrication Group（CDFG）
- **配套仓库**：<https://github.com/Frank-ZY-Dou/awesome-ai-3d-modeling-robotics>
- **对应报告**：[Frontier Models for 3D Modeling, Computational Design and Robotics](../papers/dou_frontier_3d_modeling_robotics_mit_csail_2026.md)（MIT CSAIL Research Report，living survey）
- **核对范围**：抓取并完整通读项目页当前实时版本全部正文章节（Abstract、Executive Summary、§1–§10 Materials & Methods、Empirical Benchmark & Corpus Statistics Dashboard §1–§3），涵盖从 2026-09 首版（345 帖 / 243 case / 43 可运行代码）持续更新至 **2026-11-26 快照（1,800 帖 / 755 case / 99 benchmark 套件）** 的完整演化记录。
- **通读：** 已完成

## 一句话说明

MIT CDFG 维护的 **前沿多模态模型 × 3D 建模 / 参数化 CAD / 机器人 / 动画** 实时更新实证综述站：截至 2026-11-26 版已汇总核验 **1,800** 条公开帖、**755** 个 case 与 **99** 个定量 benchmark（69 机器人与具身 + 30 3D/CAD/空间），按 **Harness 接口 · 离线/在线模式 · 反馈与可复现等级** 严格边界模型能力与工程风险。

## 开源与实时版本核对（2026-11-28 复核）

- **已开源（索引与媒体归档层）：** [awesome-ai-3d-modeling-robotics](https://github.com/Frank-ZY-Dou/awesome-ai-3d-modeling-robotics) — 归档原始帖、模型声明、媒体副本与 case 元数据；综述项目本身**不是**单一可运行训练框架或统一评测 harness。
- **规模演进对比：**
  - 2026-09 初版档案：345 公开帖 / 243 独立 case / 43 可运行代码 case。
  - **2026-11 实时页面版**：**1,800 公开帖 / 755 独立 case**（3D Modeling 458、Industrial Design & CAD 55、Robot Control 120、Animation 122）+ **99 定量 benchmark 套件**。
- **可复现分层（755 case，§10）：** **Rank 1（Code Provided）170 例** · **Rank 2（Interactive Demo/Web CAD）188 例** · **Rank 3（Demo Media Only）397 例**；过半数展示仍缺完整代码与失败次数/耗时分母。
- **四档实证标准（Empirical Dashboard §3）：** Established（独立多随机种子开源复现） / Partial（部分条件可复现或厂商技术报告） / Not Established（单条演示缺重试与成本分母） / Absent or Refuted（实物迁移失败、热保护/碰撞/安全拒答失效如 RoboHarm）。
- **核心工程边界摘要（§5–§9）：** Frontier 模型在 3D/CAD 与机器人领域呈现显著分化——擅长可编辑 Blender/CAD 初稿（如 Parametric CAD Bench 70.3% → 84.8%、SteelEnv 90.5% pass@1 但强约束 H3/H4 归零）与离线仿真写控（RoboPianist F1 0.907、Artisan/USDCraft 生成带关节 USD），而在线实机闭环受多秒延迟与接触精度限制（RoboDojo-Sim precision 仅 4%，RT-Safe 实时无 pause 碰撞激增、safe success 0%，RoboHarm 安全拒答极低且有硬件损坏中止记录）。

## 对 wiki 的映射

- [前沿模型 × 3D/CAD/机器人 living survey 阅读坐标](../../wiki/overview/frontier-models-3d-cad-robotics-survey.md)
