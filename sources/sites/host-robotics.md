# HOST 项目页

- **论文：** [arXiv:2607.20033](https://arxiv.org/abs/2607.20033)
- **项目页：** <https://host-site.host-robotics.workers.dev/>
- **代码：** <https://github.com/CGuangyan-BIT/HOST>
- **权重：** <https://huggingface.co/Guangyan/HOST>
- **机构：** 北京理工大学（BIT）；自变量机器人（X Square Robot）；清华大学（Tsinghua）
- **入库日期：** 2026-09-04
- **更新日期：** 2026-11-19
- **核对范围：** 通读项目页全页内容（Overview、50 个未见任务结果展示、八任务对照组、10/20/50 条示范技能保留对比表、光照/OOD 物体/场景替换/人为干扰逐任务成功率表与 BibTeX）。
- **通读：** 已完成

## 开源核查（2026-09-04 初查 / 2026-11-19 复核）

| 组件 | 状态 |
|------|------|
| 项目页 | 有 — 方法片、任务片、八任务对照、效率/遗忘、扰动表、BibTeX |
| 代码 | **已开源** — [CGuangyan-BIT/HOST](https://github.com/CGuangyan-BIT/HOST) |
| 权重 | **已发布** — HF `Guangyan/HOST`（MIT） |
| 真机大规模数据 | **未随仓发布** |
| 仓级 LICENSE | **未齐**（仅 `policy_training/LICENSE` 继承 Fast-WAM） |

**结论：** **已开源**（代码 + 权重）。真机 19 万轨迹与人视频配对不在公开下载清单里。

## 页上可核数字（项目页逐表）

- 习得时间平均 **29 s**；八任务成功率 **62%**；相对 $\pi_{0.5}$+SFT **507×**；已掌握技能保留 **99%**。
- 八任务分组条（均值 / 逐任务 %）：Vid2Robot **16%** · AWDA **19%** · $\pi_{0.5}$ **11%** · Wall-OSS **17%** · HOST-base **4%** · HOST **62%**（Place fruits 50 / Pick pen 55 / Stack bowls 75 / Wipe plate 60 / Insert pen 50 / Cover book 60 / Stack pots 65 / Fold socks 80）。
- 扰动相对默认 62%：光照 **61**（−1）、OOD 物体 **58**（−4）、换场景 **56**（−6）、人为干扰 **53**（−9）。
- 遗忘留存（7 任务平均保留率，10 / 20 / 50 demos）：$\pi_{0.5}$ `46% → 20% → 22%`、Wall-OSS `62% → 43% → 43%`、HOST-base `46% → 20% → 22%`，而 HOST 无需梯度更新保持原有水平（总览留存 **~99%**）。

## 交叉

- 论文归档：[`sources/papers/host_arxiv_2607_20033.md`](../papers/host_arxiv_2607_20033.md)
- 仓库归档：[`sources/repos/host.md`](../repos/host.md)
- wiki：[`wiki/entities/paper-host-one-shot-human-video.md`](../../wiki/entities/paper-host-one-shot-human-video.md)
