# SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2511.23300>
- **入库日期：** 2026-07-10
- **一句话说明：** SafeHumanoid 把「怎么调阻抗」这个低层控制问题，交给一个第一视角 VLM + RAG 检索库来回答：头部相机画面 → VLM 抽成结构化场景语义（任务、物体易碎性、是否有人、障碍等）→ 在 16 条经安全标准验证的模板库里做最近邻检索 → 取回每关节的刚度 Kp / 阻尼 Kd / 速度，下发给 50 Hz 的板载阻抗控制器。一旦画面里出现人手，机器人自动降刚度、升阻尼、减速，在不丢任务的前提下提升人机协作安全性。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod](../../wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md).

## 对 wiki 的映射

- [paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod](../../wiki/entities/paper-notebook-safehumanoid-vlm-rag-driven-control-of-upper-bod.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html>
- 论文：<https://arxiv.org/abs/2511.23300>
