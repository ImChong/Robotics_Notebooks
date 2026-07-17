# Sunday Robotics（sunday.ai）

- **类型：** 公司 / 家庭通用机器人（项目站）
- **收录日期：** 2026-07-17
- **主站：** <https://www.sunday.ai/>
- **说明：** 美国家庭机器人公司，对外品牌 **Sunday Robotics** / **「The helpful robotics company」**；自研移动平台 **Memo** 与 **ACT** 系列机器人基础模型。

## 一句话

端到端垂直整合 **Memo 硬件 + ACT 基础模型 + 机队数据环** 的家庭通用机器人路线；2026-07 预览 **ACT-2** 叠衣 **Solve**（99.1% / 零部署适配）。

## 为什么值得保留

- **产业侧「可靠性 × 泛化」叙事样本**：与学术 VLA / BFM 路线对照，强调 **人类 sensorized 预训练 + in-house post-training 可跨家庭迁移**。
- **Solve 评测框架** 对「demo 成功率」解构有方法论参考价值（Performance / Scope / Adaptation cost）。
- 硬件侧 **三指家务夹爪** 常被硬件科普文引用，与「不必 24DoF 仿人手」论点一致。

## 项目页核查（步骤 2.5 · 2026-07-17）

| 核查项 | 结论 |
|--------|------|
| **主站 Footer / 导航** | Careers、Blog、产品视频；**无 Code / GitHub / Hugging Face 入口** |
| **首页 HTML 检索** | `github` / `huggingface` / `open-source` **无匹配** |
| **开放程度** | **未开源** — 模型权重、训练代码、Memo 完整硬件 CAD/SDK 均未公开 |
| **部分开放** | 博客含 **评估视频合集**、**BibTeX**、定量表格；Beta Program 报名入口 |
| **论文/技术报告** | ACT-1 以博客/脚注引用（*ACT-1: A Robot Foundation Model Trained on Zero Robot Data*, 2025）；ACT-2 **完整技术报告待发布**（博客称 post-training 细节另文） |

- **代码：** 截至入库日 **无官方仓库链接**
- **数据集：** **未公开**（人类 sensorized 数据为 proprietary）
- **模型 checkpoint：** **未公开**

## 公开信息要点

### 产品与模型时间线（博客归纳）

| 时间 | 里程碑 |
|------|--------|
| 2025-11 | **ACT-1**：长时程移动操作、未见家庭泛化、灵巧操作；**零机器人数据** 预训练叙事 |
| 2026-03 | Series B 融资博文（同站 *Sunday's Series B: No More Demos*） |
| 2026-07 | **ACT-2 Preview**：泛化可靠性、叠衣 Solve、单示范 SFT、2026 秋 Beta |

### 硬件线索

- **Memo**：移动全身家用机器人（非固定桌面臂）；博客强调 **全身 reposition / 高度调节** 扩展叠衣工作空间。
- **末端执行器**：外部硬件科普文引用 **Sunday 三指夹持器** 作家务任务专用夹爪案例（见 [humanoid-hardware-101 传感与末端](../../wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)）。

## 交叉链接

- 博客归档：[sunday_act2_preview.md](../blogs/sunday_act2_preview.md)
- Wiki 实体：[sunday-robotics-act2.md](../../wiki/entities/sunday-robotics-act2.md)
- 概念：[robotics-solve-standard.md](../../wiki/concepts/robotics-solve-standard.md)
