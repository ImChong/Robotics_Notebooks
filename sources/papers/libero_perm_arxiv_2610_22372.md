# LIBERO-PeRM 论文来源归档

> 来源归档（ingest）

## 元数据

- **题目：** LIBERO-PeRM: Benchmarking Personalized Robotic Manipulation
- **作者：** Zhixu Li、Keqian Tang、Litian Gong、Jingyu Yao、Wenqian Zhang、Tianze Xu、Zehao Wang、Yuping Wang、Junge Zhang、Jiachen Li
- **机构：** 加州大学河滨分校、耶鲁大学、佐治亚理工学院、密歇根大学、多伦多大学
- **预印本：** <https://arxiv.org/abs/2610.22372>
- **项目页与资源入口：** <https://libero-perm.github.io/>
- **发表信息：** NeurIPS 2026 Evaluations and Datasets Track（项目页标注）
- **版本：** v1，2026-10-15
- **入库日期：** 2026-10-19
- **一句话说明：** 在 LIBERO 仿真任务中分离“完成任务”和“按用户偏好完成”，系统评测机器人策略的个性化能力。

## 核心论文摘录

### 基准设计

偏好被定义为执行方式的条件，而不是新任务目标，分为运动、技能和用户画像三类。项目页列出 37 种单偏好、18 种偏好对、五个任务套件、312 个任务及约 40.8K 条含偏好与无偏好示范。自动化生成流程为偏好场景生成配对轨迹、子任务元数据和可自动计算的偏好满足度指标。

### 评测与结果

- 评估零样本遵循、示范学习效率、跨布局/任务泛化、偏好组合与冲突处理。
- 当前 VLA 的零样本个性化较弱；监督微调使偏好变得可学习。
- 混合训练较顺序适配更能保留无偏好表现；上下文提示优于仅偏好训练，但运动偏好学习可能变弱。跨任务泛化、偏好分解和冲突处理仍是难点。

## 开源状态

项目页提供 Code 与 Dataset 下载入口，列出生成/评测代码、策略服务、rollout 客户端、BDDL 场景文件和示范。页面通过脚本处理下载目标，本次未确认稳定的独立 GitHub URL；后续可从项目页资源按钮更新具体仓库与数据集地址。

## 对 wiki 的映射

- [LIBERO-PeRM 实体页](../../wiki/entities/paper-libero-perm-personalized-manipulation.md)
- [官方项目页归档](../sites/libero-perm.md)

## 一手来源

- [arXiv 摘要](https://arxiv.org/abs/2610.22372)
- [项目页、代码与数据入口](https://libero-perm.github.io/)
