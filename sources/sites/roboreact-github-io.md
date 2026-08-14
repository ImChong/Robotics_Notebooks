# RoboReact 项目页（roboreact.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://roboreact.github.io/>
- **标题：** RoboReact — Agentic Skill Distillation from Generated Egocentric Videos
- **机构：** 香港中文大学（深圳）；京东科技；清华大学
- **论文：** <https://arxiv.org/abs/2608.03387> — 归档见 [`sources/papers/roboreact_arxiv_2608_03387.md`](../papers/roboreact_arxiv_2608_03387.md)
- **入库日期：** 2026-08-14
- **一句话说明：** 落地页展示四阶段管线（生成–编译–标定精炼–再接地执行）、四任务真机视频与 Table 1 成功率。截至入库日 **无 Code / Hugging Face / 权重下载**。

## 开源核查（步骤 2.5，2026-08-14）

| 项 | 状态 |
|----|------|
| **论文承诺** | PDF / HTML **未写** code will be released |
| **项目页 Code 区** | **无** GitHub 训练仓、Hugging Face、Zenodo、ModelScope |
| **GitHub 用户 `RoboReact`** | 仅 [`RoboReact/RoboReact.github.io`](https://github.com/RoboReact/RoboReact.github.io)（Pages 落地页） |
| **结论** | **确认未开源**。勿把落地页仓当成可复现实现；勿建 `sources/repos/`。放出训练/推理入口后再补仓库归档与论文页时序图。 |

## 页面结构速记

1. **Why** — 单帧 egocentric RGB-D + 语言指令，生成人类交互视频提供任务时序与几何结构。
2. **四阶段** — Generate & select → Recover & compile → Calibrate & refine → Re-ground & execute。
3. **视频廊** — Hand Over / Open Box / Pour Water / Open Drawer；跨物体、变位姿、蹲–操作耦合。
4. **Table 1** — 终端 SR：RoboReact 85 / 70 / 85 / 85，对齐论文。
5. **其它数字** — 15 round 标定；5.6-ultra 编辑器；Seedance 2.0 vs 1.5 Pro；冻结栈扰动保留 80–94% Avg. Len.。

## 关联资料

- 论文摘录：[`sources/papers/roboreact_arxiv_2608_03387.md`](../papers/roboreact_arxiv_2608_03387.md)
- Wiki 实体：[`wiki/entities/paper-roboreact.md`](../../wiki/entities/paper-roboreact.md)
- 低层控制器对照：[HOMIE](../../wiki/entities/paper-loco-manip-161-040-homie.md)
- 真机平台：[Unitree G1](../../wiki/entities/unitree-g1.md)
