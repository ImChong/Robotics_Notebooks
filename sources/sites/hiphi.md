# HiPHI 官方项目页与数据发布入口

- **类型：** site / dataset / motion-capture / human-object-interaction
- **项目页：** <https://noitom-robotics.github.io/hiphi/>
- **论文：** <https://arxiv.org/abs/2608.16222>（[摘录](../papers/hiphi_arxiv_2608_16222.md)）
- **代码：** <https://github.com/noitom-robotics/hiphi>（[归档](../repos/hiphi.md)）
- **数据集：** <https://huggingface.co/datasets/noitomrobotics/HiPHI>
- **在线 Viewer：** <https://hiphi-viewer.modalitynet.com/>
- **机构：** 诺亦腾机器人；新加坡国立大学；清华大学深圳国际研究生院；香港科技大学；香港大学
- **核查日期：** 2026-10-02
- **一句话说明：** 高精度人体运动与物体交互基准的展示、数据入口和 G1 下游实验汇总。

## 项目页要点

- FrameNet 用于系统设计采集；同步人体、物体位姿与几何用于 object-aware 全身模仿。
- 617.5 h 发布量含镜像增强，原始采集约 308.7 h；HOI 占发布时长 39.8%。
- 项目页展示覆盖、质量、数据扩展和 G1 真机行为，在线 Viewer 可浏览运动与元数据。

## 源码与数据开放核查

| 资源 | 截至核查日的边界 |
|------|------------------|
| GitHub | **部分开放**：项目网站、格式文档、本地 Motion Viewer；未找到完整训练/部署管线与策略权重 |
| Hugging Face 数据 | **已发布、门控访问**：填写申请并同意许可后下载，非免登录无限制获取 |
| 数据许可 | ModalityNet Open Research License v1.0，面向非商业研究、教育与评测；商业许可另行联系 |
| 本地 Viewer | viewer/README.md 与 viewer/LICENSE 标注 Apache-2.0；不覆盖数据许可 |

## 对 wiki 的映射

- [HiPHI 论文与数据集实体](../../wiki/entities/paper-hiphi.md)
- [人形参考运动与操作数据集选型](../../wiki/comparisons/humanoid-reference-motion-datasets.md)
