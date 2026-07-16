# Xiaomi-Robotics-1 项目页与技术报告

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories
- **类型：** site（品牌站项目页 + PDF 技术报告；代码/权重待发布）
- **组织：** Xiaomi Robotics（小米机器人实验室）
- **品牌站说明页：** <https://robotics.xiaomi.com/xiaomi-robotics-1.html>
- **技术报告 PDF：** <https://robotics.xiaomi.com/robot-static-resource/xiaomi-robotics-1/xiaomi-robotics-1.pdf>
- **入库日期：** 2026-07-16
- **一句话说明：** 小米 **具身基座 VLA**：**10 万小时** embodiment-free **UMI** 轨迹预训练（**>1,700** 场景、VLM **自动状态转移标注**）+ **~1 万小时** 跨本体后训练（**>7,200h** 真机移动操作/双臂 + 开源数据集）；**Qwen3-VL + DiT flow matching + Choice Policies** 的 **MoT** 架构（**2B / 5B / 10B**）；预训练 **数据/模型 scaling** 可预测地迁移到未见环境真机开箱成功率；少样本微调四任务平均 **75%**（**<10h/任务**）超 **π₀.₅ 40%**；**RoboCasa / RoboCasa365 / VLABench / RoboDojo** 四基准 SOTA。
- **沉淀到 wiki：** [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **100k h UMI 预训练 + 跨本体后训练** 的 scaling 实证；与 **π₀.₅ / XR-0** 对照 |
| [Teleoperation](../../wiki/tasks/teleoperation.md) | 预训练走 **无机器人 UMI**；后训练含 **移动操作 + 双臂** 真机示教 |
| [Manipulation](../../wiki/tasks/manipulation.md) | 开箱移动操作、鞋柜/沙发/厨房整理；少样本 phone packing / laundry 等 |
| [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) | 同族 **Qwen3-VL + DiT + Choice Policies**；XR-0 强调 **异步实时部署**，XR-1 强调 **超大规模 UMI 预训练 scaling** |
| [Xiaomi-Robotics-U0](../../wiki/entities/xiaomi-robotics-u0.md) | 同实验室 **38B 合成 WM** 与 **XR-1 基座 VLA** 形成数据—策略分工 |

---

## 设计要点（官网 / PDF 归纳）

### 1) 数据

- **预训练：** **>100k h** 真实世界 **UMI** 手持夹爪 + egocentric 相机轨迹，覆盖家庭/商业/工业/办公/户外，**>1,700** 场景。
- **自动标注：** 轨迹切 **等长片段** → **Qwen3.5-27B** 描述 **夹爪与交互物体的状态转移**（非 imperative 指令）；producer–consumer 管线约 **两周** 标完全库。
- **后训练：** 合计 **~10k h** 跨本体数据——**>7,200 h** 自采移动操作/双臂真机（沙发整理、鞋柜、厨房收纳等）+ **>1k h** 人工分段 **imperative 指令** UMI + 过滤后的 **Bridge V2 / RT-1 / DROID** 等开源集。

### 2) 模型

- **MoT：** **Qwen3-VL** + 同层数更小 hidden 的 **DiT**，**flow matching** 生成 action chunk；**Choice Policies** 在 VLM 侧辅助收敛，但 **DiT 注意力排除 VLM action token** 以防抄近路。
- **规模：** **2.6B / 5.1B / 10.5B** 三档（VLM **2.1B / 4.4B / 8.8B** + DiT **470M / 604M / 1.5B**）。
- **预训练目标：** \(L = L_{\text{Flow}} + L_{\text{Regression}} + 0.1 \cdot L_{\text{NTP}}\)；VL : UMI = **1:9**；每样本 **4** 个 flow 时间步摊销 VLM 成本。

### 3) Scaling 与评测

- **预训练：** 在 **~20k h** UMI 子集上，验证 **MSE** 随数据从 **12.5%→100%**、模型 **2B→10B** 单调下降。
- **后训练开箱（未见环境/物体）：** 预训练数据越多、模型越大，四任务（鞋柜/书包/桌面/沙发）总成功率越高（**100% 20k h + 10B → ~79%** vs 无 action 预训练 **26%**）。
- **少样本微调：** 四灵巧任务平均 **<10 h/任务** 总成功率 **75%** vs **π₀.₅ 40%**；**<40 h/任务** **85%** vs **53%**。
- **仿真：** RoboCasa **74.5%**、RoboCasa365 **57.4%**、VLABench **59.1%**、RoboDojo **13.93**（官网表；PDF 细节略异）。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/xiaomi-robotics-1.md`**：数据—训练—scaling—下游适配实体页（双 Mermaid：预训练管线 / 后训练对齐）。
- 更新 **`wiki/entities/xiaomi-robotics-0.md`**、**`wiki/methods/vla.md`**：小米 VLA 谱系交叉引用。

---

## 外部参考

- Xiaomi Robotics, *Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Data*, arXiv preprint（项目页 BibTeX，编号待公开）
- [Robotics @ Xiaomi — XR-1](https://robotics.xiaomi.com/xiaomi-robotics-1.html)
- [技术报告 PDF](https://robotics.xiaomi.com/robot-static-resource/xiaomi-robotics-1/xiaomi-robotics-1.pdf)
