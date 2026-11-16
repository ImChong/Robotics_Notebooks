# sylvestf.github.io/wcm-homepage — WCM 项目主页

> 项目官方网站归档

- **URL：** <https://sylvestf.github.io/wcm-homepage/>
- **对应论文：** [arXiv:2607.29613 — World Critic Model](../papers/wcm_world_critic_arxiv_2607_29613.md)
- **对应代码：** [sylvestf/WCM](../repos/wcm-world-critic-model.md)
- **通读：** 已完成
- **核对范围：** 已通读 `originals/sites/wcm.html` 全文（页首机构、摘要与核心视频/可视化展示区、资源链接矩阵、引用信息与底部更新状态声明）。

---

## 站点资源与开源核实摘要

1. **作者与机构声明：** Senyu Fei\*、Zihao Wu\* 等来自同济大学（Tongji University）、上海创智学院（Shanghai Innovation Institute）与复旦大学（Fudan University）。
2. **官方外链清单：**
   - **Paper (arXiv)：** <https://arxiv.org/abs/2607.29613>
   - **Code (GitHub)：** <https://github.com/sylvestf/WCM>
   - **Model & Dataset (Hugging Face Collection)：** <https://huggingface.co/collections/Sylvest/wcm>
3. **关键可视演示内容：**
   - **结构对比图（Main Figure & Architecture）：** 展示单帧无记忆 Critic 在 OOD/停滞时的失败案例，以及 WCM 通过动作隔离上下文骨干与动作条件动力学预测解决 POMDP 错配的架构。
   - **真机七任务与 Sim-to-Real 视频对比：** WidowX-250S 上的萝卜/香蕉/青椒抓取、布料与毛巾折叠、灶台擦拭、传送带寿司分拣等 $\pi_{0.5}$+WCM vs RECAP/AWR 对比，以及 `episode_value_video` 生成的成功/失败片段实时价值曲线变化（成功推进稳定上升，抓取打滑或空抓阶段急剧跌落）。
4. **开源状态同步（与 GitHub README 一致）：** 主页提供的 Code 与 Hugging Face 链接均真实有效；当前公开 Critic 训练/评测/视频渲染代码及 Pick-and-Place、LIBERO-Plus 部分数据与 Critic 权重，完整 RL 微调管线标注持续开源中（部分开源）。
