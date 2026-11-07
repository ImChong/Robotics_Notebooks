# SmellDiffusion（arXiv:2609.20624）

> 来源归档（paper）；升级既有条目，不重复建页。

- **标题：** SmellDiffusion: Diffusion-Based Quadruped Navigation with Olfactory Scene Graphs
- **类型：** paper
- **作者：** Faith Ogunwoye、Iana Zhura、Hajira Amjad、Timofei Kozlov、Didar Seyidov、Dmitrii Plotnikov、Fedor Fedorov、Dzmitry Tsetserukou
- **机构：** 斯科尔科沃科学技术学院（Skoltech）
- **arXiv：** <https://arxiv.org/abs/2609.20624v2>
- **全文：** <https://arxiv.org/html/2609.20624v2>
- **PDF：** <https://arxiv.org/pdf/2609.20624v2>
- **版本：** v2，2026-09-18；首次提交 2026-09-17
- **入库日期：** 2026-09-20
- **核对日期：** 2026-11-07
- **核对范围：** v2 §I–IX、Fig. 1–5、Table I–VII 与参考文献；HTML 全文无独立附录。核对机制、数据切分、规划计时、语言泛化、Gazebo 执行及局限。
- **通读：** 已完成

## 开放状态核查

已读 arXiv 页面与全文未列官方项目页或可获取的代码/artifact URL。第三方线索所指 `https://github.com/FaithOgunwoye/Olfsg` 经 GitHub API 与浏览器直接打开均为 404（2026-11-07）；仅记录访问结果，不确认为官方仓库，不推断其私有/删除原因。论文 §VIII 描述 revision artifact 内容，但描述不等于可下载验证。

## 核心摘录

1. 图谱保存气体身份与目标；离线物种独立 GADEN 场和固定高度虚拟扫掠，不是行走中的主动嗅觉探索。
2. 峰值局部门控仅在 14/204 评估配置调用训练库匹配，偏移子集误差降 60%；全量匹配使整体误差恶化。
3. 相同目标的四种规划器均通过 12 个请求；普通 A* 最快最短，扩散十选一提高羽流暴露但顺序计算耗时约 408 ms。
4. 新语言查询仅 3/10，候选相关 NNN 后 5/10；真实混合气体、在线流场和物理四足均待验证。
5. Gazebo 仅每气体/规划器一次执行；到源均值差异只能描述，不能推出显著优势。扩散执行继续依赖 A* 修复。

## 对 wiki 的映射

- [SmellDiffusion：方法、评测与工程边界](../../wiki/entities/paper-smelldiffusion.md)
