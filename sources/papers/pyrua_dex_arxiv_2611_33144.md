# PyRUA-Dex: Learning Programmatic Policies for Dexterous Bimanual Manipulation with Coding Agents

- **类型：** paper
- **论文：** [摘要](https://arxiv.org/abs/2611.33144v1) · [PDF](https://arxiv.org/pdf/2611.33144v1) · [HTML](https://arxiv.org/html/2611.33144v1)
- **版本：** v1，2026-11-24。
- **作者：** Feixiang Ruan、Guodong Lv、Kewei Zhang、Wenbo Huang、Yintianrun Zhang、Qiang Wang、Ruiyang Si、Shunyu Yang、Shulong Jiang、Weicheng Duan、Rui Ni、Jianxin Bi、Lekai Wang、Yuze Fan、Haobo Li、Zeliang Fu、Jiankun Zhang、Duomin Wang、Enze Xie、Daquan Zhou。
- **机构：** 北京大学、新加坡国立大学、英伟达、同济大学、芝加哥大学。
- **项目页：** <https://vme0522.github.io/PyRUA-Dex/>
- **官方分支（代码待公开）：** <https://github.com/DAGroup-PKU/PyRUA-Lean/tree/dex>
- **核对范围：** v1 正文 §1–6；附录 A 的基准与测试条件、B 的阶段程序和控制原语、C 的补充消融与成本、D 的真机辨识和随机化、E 的复现索引。
- **通读：** 已完成
- **对应节点：** [PyRUA-Dex](../../wiki/entities/paper-pyrua-dex.md)

1. §3：开发分探索和精炼，运行分阶段的冻结程序，不需要在线语言模型参与。
2. §4 与附录 C：Bench2Dex 综合分 72.9，对应 60.6% SR；DexJoCo 完整随机化为 81.0% SR。开发资源与策略部署期资源需独立汇报。
3. §5–6 与附录 D：辨识关节驱动以缩小仿真差异，使用真机反馈更新仿真；真机为示范，可靠性数字来自标定仿真。

[项目归档](../sites/pyrua-dex.md) · [官方分支归档](../repos/pyrua-dex.md)
