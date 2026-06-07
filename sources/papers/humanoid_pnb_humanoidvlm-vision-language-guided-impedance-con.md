# HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2601.14874>
- **入库日期：** 2026-06-07
- **一句话说明：** HumanoidVLM 把"挑阻抗参数 + 选抓取角"这件老靠手调的事，外包给一个轻量管线：VLM 看一眼第一视角图把任务和物体说出来 → FAISS-RAG 从两个小数据库（9 个任务 + 9 个物体）里查出实验验证过的 stiffness/damping 与手指角→ 直接喂给 G1 的任务空间阻抗控制器，让接触富集的人形操作"软硬合适"。14 个测试场景命中率 93%。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-humanoidvlm-vision-language-guided-impedance-con](../../wiki/entities/paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md).

## 对 wiki 的映射

- [paper-notebook-humanoidvlm-vision-language-guided-impedance-con](../../wiki/entities/paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2601.14874>
