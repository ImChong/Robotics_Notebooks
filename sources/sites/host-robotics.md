# HOST 项目页

- **论文：** [arXiv:2607.20033](https://arxiv.org/abs/2607.20033)
- **项目页：** <https://host-site.host-robotics.workers.dev/>
- **代码：** <https://github.com/CGuangyan-BIT/HOST>
- **权重：** <https://huggingface.co/Guangyan/HOST>
- **机构：** 北京理工大学（BIT）；自变量机器人（X Square Robot）；清华大学（Tsinghua）
- **入库日期：** 2026-09-04

## 开源核查（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| 项目页 | 有 — 方法片、任务片、八任务对照、效率/遗忘、扰动表、BibTeX |
| 代码 | **已开源** — [CGuangyan-BIT/HOST](https://github.com/CGuangyan-BIT/HOST) |
| 权重 | **已发布** — HF `Guangyan/HOST`（MIT） |
| 真机大规模数据 | **未随仓发布** |
| 仓级 LICENSE | **未齐**（仅 `policy_training/LICENSE` 继承 Fast-WAM） |

**结论：** **已开源**（代码 + 权重）。真机 19 万轨迹与人视频配对不在公开下载清单里。

## 页上可核数字（非 PDF）

- 习得时间平均 **29 s**；八任务成功率 **62%**；相对 \(\pi_{0.5}\)+SFT **507×**；已掌握技能保留 **99%**。
- 八任务分组条：Vid2Robot 16% · AWDA 19% · \(\pi_{0.5}\) 11% · Wall-OSS 17% · HOST-base 4% · HOST 62%。
- 扰动相对默认 62%：光照 61、OOD 物体 58、换场景 56、人为干扰 53。

## 交叉

- 论文归档：[`sources/papers/host_arxiv_2607_20033.md`](../papers/host_arxiv_2607_20033.md)
- 仓库归档：[`sources/repos/host.md`](../repos/host.md)
- wiki：[`wiki/entities/paper-host-one-shot-human-video.md`](../../wiki/entities/paper-host-one-shot-human-video.md)
