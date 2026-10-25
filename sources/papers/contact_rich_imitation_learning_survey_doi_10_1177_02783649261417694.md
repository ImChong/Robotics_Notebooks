# A survey on imitation learning for contact-rich tasks in robotics

- **类型：** paper
- **来源：** SAGE Journals
- **期刊：** The International Journal of Robotics Research（OnlineFirst，2026）
- **首次在线发表：** 2026-03-14
- **作者：** Toshiaki Tsuji、Yasuhiro Kato、Gokhan Solak、Heng Zhang、Tadej Petrič、Francesco Nori、Arash Ajoudani
- **DOI：** <https://doi.org/10.1177/02783649261417694>
- **原文：** <https://journals.sagepub.com/doi/10.1177/02783649261417694>
- **开放许可：** Creative Commons Attribution-NonCommercial 4.0（CC BY-NC 4.0）
- **一句话说明：** 系统梳理接触丰富机器人任务中的模仿学习，覆盖示教采集、传感模态、学习范式、应用领域与开放挑战。

## 核心内容

- **问题背景：** 接触动力学具有非线性且对细小位姿偏差敏感；真实机器人试错成本高，接触状态和作用力又常被视觉遮挡，因此需要从专家演示中学习细致的动作与力调节。
- **示教采集：** 讨论本体感知、力/力矩、视觉和触觉等模态，以及手把手引导、遥操作、VR 示教和观察式示教；并讨论合成数据与可用数据集。
- **学习方法：** 将行为克隆、动态运动基元、VAE/扩散等生成方法、基础模型、逆强化学习、多模态模仿学习、离线强化学习及其他新方法放在接触操作视角下整理。
- **应用领域：** 归纳工业、家庭与医疗等场景，并强调不同任务对传感和算法选择的约束。
- **挑战：** 数据效率与泛化、接触动力学和 Sim2Real、触觉数据采集、视觉遮挡，以及第三人称示教到机器人第一人称动作的视角/外观差距。

## 论文边界

这是综述论文，贡献是系统分类与研究脉络整理，不提出新的机器人系统或统一定量基准。SAGE 文章页未列独立项目主页、代码仓库或数据集入口；复现实验需回到综述引用的原始研究。

## 对 wiki 的映射

- [论文实体页](../../wiki/entities/paper-imitation-learning-contact-rich-tasks-survey.md) — 整理研究分类、工程读法与局限。
- [Imitation Learning](../../wiki/methods/imitation-learning.md) — 补入接触任务中的示教与模仿学习综述。
- [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md) — 连接接触感知、示教与学习方法。

## 参考

- Tsuji et al., “A survey on imitation learning for contact-rich tasks in robotics,” *The International Journal of Robotics Research*, first published online 2026-03-14. DOI: 10.1177/02783649261417694.