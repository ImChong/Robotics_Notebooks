# CondVLN 官方项目页核查

> 来源归档（核查日期：2026-10-13）

- **资料类型：** official project page / project-link audit
- **项目页：** <https://condvln.github.io/>
- **站点源码仓：** <https://github.com/condvln/condvln.github.io>
- **论文：** [arXiv:2608.17318](https://arxiv.org/abs/2608.17318)
- **研究基准：** CondVLN
- **一句话说明：** 论文与作者发表页将该地址列作项目页；站点的 Code、Dataset 按钮指向匿名 4open artifact，而非可直接确认身份的公开 GitHub 算法仓。

## 页面与链接核查

| 项目 | 核查结果 |
|------|----------|
| 项目页身份 | arXiv v1 元数据列出该项目页；作者公开发表页也链接到该页面 |
| 代码入口 | 页面源码中标为 **Code**，指向 <https://anonymous.4open.science/r/condvln_bench/README.md> |
| 数据入口 | 页面源码中标为 **Dataset**，指向 <https://anonymous.4open.science/r/condvln_bench/CondVLN_dataset> |
| GitHub 仓库 | <https://github.com/condvln/condvln.github.io> 是项目页静态站点源码；不是由当前可见文件树证明的模型训练/评测实现 |
| 镜像内容 | 本次通过可用检索入口未能读取匿名镜像 README / 数据文件，因此代码能否直接运行、数据包含哪些文件、许可与镜像期限均未确认 |

因此，项目页确实公开给出代码与数据链接，但不把它们写成“已验证可复现”或“数据集已可下载”。链接目前是匿名评审镜像，后续若出现解匿名后的官方 GitHub / 数据页，应更新同一 CondVLN 实体，不拆出新的项目节点。

## 对 wiki 的映射

- [CondVLN 论文实体](../../wiki/entities/paper-condvln-conditional-branching.md)
- [论文原文归档](../papers/if_then_otherwise_condvln_2608_17318.md)
- [匿名代码与数据入口核查](../repos/condvln-benchmark-artifact.md)
