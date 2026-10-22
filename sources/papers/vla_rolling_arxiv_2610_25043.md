# VLA-Rolling 论文来源归档

## 元数据

- **题目：** VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse
- **作者：** Sicheng He, Yi Pan, Qiuting Yu, Jiaming Huang, Qi Lu, Shuyan Wu, Hongyan Feng, Xuanyu Liu, Jun Zhang, Jie Zhang, Wenqi Zhang, Xuhong Zhang, Jianwei Yin
- **单位：** OmniAI Group of ZJU ACES Lab, Zhejiang University（据项目页）
- **预印本：** https://arxiv.org/abs/2610.25043（v1 2026-10-18；v3 2026-10-21；arXiv 列类为 cs.AI）
- **项目页：** https://chenbao123ww.github.io/VLA-Rolling/
- **代码：** https://github.com/ZJU-OmniAI/VLA-Rolling（项目页列出；本次未访问该仓库，仓库内容未核实）
- **模型：** https://huggingface.co/Chenbao77/VLA-Rolling（项目页列出；未核实）
- **入库日期：** 2026-10-22

## 论文要解决的问题

Flow-matching 类 VLA 每次生成一段动作 chunk，却通常丢弃未执行的尾部。作者发现这段尾部保留了有用的去噪进度，从而有机会减少去噪步数。

## 方法要点（据摘要与项目页）

- 保持底层 VLA 冻结，复用未执行的动作尾部。
- 机制：继承（inheritance）、逐动作置信度估计、置信度引导的重加噪与去噪。

## 报告的结果（据摘要与项目页）

- 推理吞吐最高 1.48× 加速；最大 chunk 延迟降低 32.52%（MolmoAct2，LIBERO）。
- 成功率最高提升 15.08%（SmolVLA，LIBERO-Plus）。
- π0.5、SmolVLA、X-VLA、MolmoAct2 × 四个基准的 16 个组合中：延迟均降低；成功率在 14 个组合中提升。
- 真机（AgileX PiPER，π0.5）：成功率 70% → 75%，每 chunk 延迟 227.28 → 176.50 ms。

## Robotics_Notebooks 映射

- 实体页：[VLA-Rolling](../../wiki/entities/paper-vla-rolling.md)
- 代码与模型（项目页列出）：[VLA-Rolling 代码与模型](../repos/vla_rolling.md)

## 一手来源

- arXiv 摘要页与项目页（经 WebFetch 摘要）
