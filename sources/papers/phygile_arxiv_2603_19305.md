# PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking（arXiv:2603.19305）

> 来源归档（ingest）

- **标题：** PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking
- **类型：** paper / humanoid / text-to-motion / diffusion / motion-tracking / GMT / MoE
- **arXiv abs：** <https://arxiv.org/abs/2603.19305>
- **项目页：** <https://baojch.github.io/phygile-page/>
- **机构：** 西北工业大学；上海人工智能实验室；中国科学技术大学；清华大学；复旦大学；字节跳动；东北大学（以项目页作者脚注为准）
- **入库日期：** 2026-06-19
- **一句话说明：** 用 **物理前缀引导的机器人原生扩散生成** 与 **GMT 通才跟踪器** 闭环，在 **262 维机器人骨骼空间** 从文本直接生成可执行高动态全身动作，避免人体 text-to-motion → 重定向带来的物理不可行；真机验证 breakdance、侧手翻、高踢、180°/360° 旋跳等敏捷动作。

## 核心论文摘录（MVP）

### 1) 人体 text-to-motion → 重定向的物理鸿沟

- **链接：** <https://arxiv.org/abs/2603.19305> Abstract
- **摘录要点：** 现有 text-to-motion 多在 **人体动捕数据** 上训练，先验隐含人体生物力学、驱动、质量分布与接触策略；重定向到人形后虽可满足 **关节限位与姿态连续** 等几何约束、看起来运动学合理，但常 **违反真机执行所需的物理可行性**（动力学、接触、扭矩等）。
- **对 wiki 的映射：**
  - [PhyGile](../../wiki/entities/paper-phygile.md) — 问题定义：为何需要 robot-native 生成而非 human retarget

### 2) GMT：课程式 MoE 跟踪器 + 无标注后训练

- **链接：** <https://baojch.github.io/phygile-page/> Method（左）
- **摘录要点：**
  - **两阶段 MoE tracker**：先以 **课程约束路由** 诱导专家分工，再以 **全局软后训练 + 动态专家扩展** 吸收持续困难动作。
  - **后训练**：在 **无标注运动数据** 上进一步提升对大规模机器人运动的鲁棒性。
  - 为后续 physics-prefix 适配提供 **可验证、可微调** 的 GMT 执行底座。
- **对 wiki 的映射：**
  - [PhyGile](../../wiki/entities/paper-phygile.md) — GMT 子系统与 MoE 课程叙事
  - [paper-notebook-gmt](../../wiki/entities/paper-notebook-gmt.md) — 与独立 GMT 论文实体交叉索引

### 3) TP-MoE 条件机器人原生扩散生成（262D）

- **链接：** <https://baojch.github.io/phygile-page/> Method（右）
- **摘录要点：**
  - **TP-MoE–conditioned** 扩散策略，从 **文本** 生成 **262 维机器人骨骼空间** 运动序列。
  - **推理时** 做 physics-prefix-guided **robot-native** 生成，**消除推理期重定向伪影**，缩小 **生成–执行** 差异。
- **对 wiki 的映射：**
  - [PhyGile](../../wiki/entities/paper-phygile.md) — 生成器结构与 robot-native 空间设计
  - [扩散运动生成](../../wiki/methods/diffusion-motion-generation.md) — 控制环内生成式参考家族

### 4) Physics-prefix 适配与闭环仿真精炼

- **链接：** <https://baojch.github.io/phygile-page/> Method（中）
- **摘录要点：**
  - **可执行运动前缀** 与 **新生成 1 秒延续** 拼接，经 **预训练 GMT** 验证可行性。
  - **闭环仿真精炼** 强化动态可行性，提升 **生成动作与可跟踪动作** 的一致性。
  - **Physics-prefix 适配阶段**：在 physics-derived prefix 下用生成目标 **进一步微调 GMT**，使真机可稳定执行复杂敏捷动作。
- **对 wiki 的映射：**
  - [PhyGile](../../wiki/entities/paper-phygile.md) — Mermaid 闭环管线与 fine-tuning 阶段

### 5) 离线与真机高动态结果

- **链接：** <https://baojch.github.io/phygile-page/> Results
- **摘录要点：**
  - 站点展示 **生成动作 → 微调后动作 → 真机部署** 三段对比。
  - 真机案例：**breakdance spin**、**forward/backward cartwheel**、**high kick**、**180°/360° spin jump**；另有 crawl、frog jump、monkey、hop、punch、kneel 等多样敏捷动作。
  - 论文主张将 text-driven humanoid control 推进到 **远超行走与低动态** 的敏捷全身运动前沿。
- **对 wiki 的映射：**
  - [PhyGile](../../wiki/entities/paper-phygile.md) — 实验与真机定性证据
  - [人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md) — 「文本→机器人原生生成 + GMT」分支

## 对 wiki 的映射（汇总）

- [paper-phygile.md](../../wiki/entities/paper-phygile.md) — 主沉淀页
- 交叉更新：[diffusion-motion-generation.md](../../wiki/methods/diffusion-motion-generation.md)、[humanoid-motion-tracking-method-selection.md](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 引用（项目页 BibTeX）

```bibtex
@article{bao2026phygile,
  title={PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking},
  author={Bao, Jiacheng and Yang, Haoran and Xin, Yucheng and Liu, Junhong and Xu, Yuecheng and Liang, Han and Han, Pengfei and Ma, Xiaoguang and Wang, Dong and Zhao, Bin},
  journal={arXiv preprint arXiv:2603.19305},
  year={2026}
}
```
