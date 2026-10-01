# Grounded-Action-Model（官方 GitHub）

> 来源归档（repo）

- **标题：** Grounded Action Model: 3D Grounding as a Foundation for Robotics
- **类型：** repo
- **组织 / 维护者：** GehaoZhang6
- **链接：** <https://github.com/GehaoZhang6/Grounded-Action-Model>
- **论文：** [arXiv:2609.23863](https://arxiv.org/abs/2609.23863)
- **项目页：** <https://grounded-action-model.github.io/>
- **入库日期：** 2026-10-01
- **一句话说明：** 论文官方 GitHub；README 声明训练/推理、权重与真机配置 **即将发布**，当前为项目说明 + 引用 + 演示素材。
- **沉淀到 wiki：** [`wiki/entities/paper-grounded-action-model-3d-grounding.md`](../../wiki/entities/paper-grounded-action-model-3d-grounding.md)

---

## 开源状态（步骤 2.5，2026-10-01）

| 项 | 核查结论 |
|----|----------|
| **仓库可见性** | 公开；description 含 `(code coming soon)` |
| **可运行入口** | README **无** `train.py` / `eval.py` / 安装步骤；仅 Citation 与 teaser |
| **权重** | **未发布** |
| **结论** | **待发布** — 占位仓 + 发布预告；复现须等官方 push 代码与 checkpoint |

---

## 预期复现路径（代码发布后需回写本页）

| 阶段 | 预期模块（对齐论文 Fig. / 正文） |
|------|----------------------------------|
| Grounding | 冻结 **WildDet3D**；语言经 Flan-T5 span tagging 抽对象短语 |
| Tokens | Image adapter（16×16 网格 + 目标/机械臂 mask）+ Detection adapter（点云 encoder φ）+ state history |
| Action | **MM-DiT**（12 block）+ **flow matching** 预测长度 **H** 的 joint-position + gripper chunk |
| 部署 | 独立闭环 **或** 作为 Molmo2 等规划器的低层控制器 |

代码落地后应更新 [`wiki/entities/paper-grounded-action-model-3d-grounding.md`](../../wiki/entities/paper-grounded-action-model-3d-grounding.md) 的「源码运行时序图」节。
