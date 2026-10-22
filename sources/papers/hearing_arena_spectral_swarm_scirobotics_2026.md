# Hearing the shape of an arena with spectral swarm robotics

> 来源归档（ingest）

- **标题：** Hearing the shape of an arena with spectral swarm robotics
- **类型：** paper / swarm robotics / spectral methods / distributed sensing
- **期刊：** Science Robotics, Vol. 11, Issue 119（2026-10-21）
- **DOI：** <https://doi.org/10.1126/scirobotics.adr9578>
- **预印本：** <https://arxiv.org/abs/2403.17147>（2024-03-25）
- **作者：** Leo Cazenille、Nicolas Lobato-Dauzier、Alessia Loi、Mika Ito、Olivier Marchal、Nathanael Aubert-Kato、Nicolas Bredeche、Anthony J. Genot
- **机构：** 巴黎西岱大学与 CNRS、索邦大学与 CNRS、东京大学、御茶水女子大学、让·莫内大学与 CNRS 等（按 arXiv 预印本作者单位）
- **项目页 / 代码：** arXiv 条目未列项目页或代码仓库
- **入库日期：** 2026-10-23
- **一句话说明：** 让稀疏 Kilobot 群体仅通过邻居间的信息扩散近似计算 Laplacian 谱，从而一次性识别机器人所处 arena 的形状，并揭示群体规模与局部交互半径之间的普适尺度关系。

## 摘要级要点

- 论文将连续空间上的 Laplacian 扩散过程映射到局部机器人交互，使群体无需单体全局地图即可估计 arena 的谱特征。
- 通过稀疏 Kilobot 群体实验验证了 arena 形状的一次性分类；核心信号是局部扩散动力学所涌现的全局谱信息。
- 作者报告最优机器人数量（全局量）与最优交互半径（局部量）之间存在普适尺度关系。
- 该思路可用于需要群体形成环境共识的任务，例如未知地形适应、分工与 quorum sensing；这些是潜在应用方向，不等同于本文已验证的实验结果。

## 核心论文摘录

### 1) 分布式局部交互恢复全局几何线索

- **链接：** [Science Robotics DOI](https://doi.org/10.1126/scirobotics.adr9578)；[arXiv 预印本](https://arxiv.org/abs/2403.17147)
- **核心贡献：** 机器人通过与邻居扩散信息以仿真 Laplacian 算子，再由其谱特征推断 arena 几何；论文在稀疏 Kilobot 群体上演示一次性形状分类。
- **对 wiki 的映射：** 群体机器人分布式感知、谱图方法与 Laplacian 特征；本次先保留为来源记录，未新增 wiki 页。

## 资料核对说明

arXiv 页面确认了预印本题名、作者名单与摘要内容。期刊卷期、日期和正式 DOI 按用户提供的信息记录；Science DOI 落地页在本次检索环境中无法读取。预印本作者单位包括巴黎西岱大学 / CNRS、索邦大学 / CNRS、东京大学、御茶水女子大学及让·莫内大学 / CNRS；未将东京大学误写为东京工业大学。
