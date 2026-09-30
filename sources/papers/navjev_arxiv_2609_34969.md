# NavJev（arXiv:2609.34969）

> 来源归档（ingest）

- **标题：** NavJev: Efficient Vision-Language Navigation via Action-Centric Visual Compression and Discriminative Action-Semantic Memory
- **缩写：** **NavJev**
- **类型：** paper / vision-language-navigation / system-one
- **arXiv：** <https://arxiv.org/abs/2609.34969>
- **项目页：** <https://kai-sheng-caesar.github.io/NavJev/>
- **代码：** 项目页与 arXiv 页 **未列公开 GitHub**（截至 2026-09-30）
- **作者：** Kai Sheng, Liuyi Wang†, Jinlong Li, Haojie Dai, Chengju Liu, Qijun Chen†
- **机构：** 同济大学（Tongji University）电子与信息工程学院
- **入库日期：** 2026-09-30
- **一句话说明：** **零样本 VLN-CE** 框架：用 **ACVC** 把 waypoint 几何 + **BLIP**  caption + **RAM** 标签压成 **动作中心文本 state**，**DASM** 滤除共享语义、保留判别性动作证据，再由 **Jev** 做 **typed waypoint/STOP 选择**；R2R-CE 上 **27.0% SR / 22.4% SPL**，**0.65 s/步**，显著低于代表性 MLLM VLN 的步级延迟与成本。

## 核心论文摘录（MVP）

### 1) 问题与范式转换（Abstract）

- **链接：** <https://arxiv.org/abs/2609.34969>
- **核心贡献：** 逐步 **自回归 MLLM 推理** 与 VLN 实际需要的 **有限 waypoint 选择** 不匹配；NavJev 将在线导航改为 **视觉压缩 + 轻量 typed 决策**。
- **对 wiki 的映射：**
  - [NavJev 论文实体](../../wiki/entities/paper-navjev-efficient-vln-jev.md)
  - [Jev（TypeSafe）](../../wiki/entities/typesafe-jev.md)

### 2) ACVC + DASM + Jev（Method）

- **链接：** 项目页 Method §01–03
- **核心贡献：**
  - **ACVC：** 每个候选 waypoint → 方向/距离 + BLIP 描述 + RAM 语义 tag。
  - **DASM：** 去掉邻域动作共享 tag，保留 **动作专属** 语义与历史。
  - **Jev：** 在约束 option label 上 **直接概率选择** waypoint 或 STOP。
- **对 wiki 的映射：**
  - [NavJev 论文实体](../../wiki/entities/paper-navjev-efficient-vln-jev.md)

### 3) R2R-CE 主结果（Table 1–2）

- **链接：** 项目页 Evaluation
- **核心贡献（零样本行）：** NavJev **NE 7.48, OSR 35.0, SR 27.0, SPL 22.4**；步时 **0.65 s**，峰值 GPU **4.46 GiB**；相对 P2DNav（50% SR）精度更低但 **~7.6× 更快**（4.92 s vs 0.65 s/步）。
- **对 wiki 的映射：**
  - [视觉–语言导航任务](../../wiki/tasks/vision-language-navigation.md)

### 4) 消融与真机（Table 4–5）

- **链接：** 项目页 Table 4–5
- **核心贡献：** BLIP+RAM+DASM 全开达 **SR 27.0 / SPL 22.4**；办公室+咖啡厅 **20 任务** 真机 NavJev **平均 SR 50%** vs Qwen3.8-Max **35%**，决策延迟 **0.65 s vs 1.10 s**。
- **对 wiki 的映射：**
  - [NavJev 论文实体](../../wiki/entities/paper-navjev-efficient-vln-jev.md)

## 参考链接

- arXiv：<https://arxiv.org/abs/2609.34969>
- 项目页：<https://kai-sheng-caesar.github.io/NavJev/>
