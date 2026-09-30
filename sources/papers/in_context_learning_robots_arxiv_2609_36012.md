# In-Context Learning for Robots: Methods and Applications（arXiv:2609.36012）

> 来源归档（ingest）

- **英文标题：** In-Context Learning for Robots: Methods and Applications
- **标题：** 机器人 In-Context Learning：方法与应用（综述）
- **类型：** paper（survey）
- **作者：** Haojian Huang, Zexi Li, Junhao Guo, Yehang Zhang, Wenxuan Peng, Bohan Zhou, Weilin Ruan, Leyi Wu, Chenxu Wang, Jianchong Su, Binghui Xie, Wosong Chen, Yingjie Xu, Tianhao Zhou, Suzeyu Chen, Pukun Zhao, Jiaqi He, Xinyi Li, Runze Li, Peiran Dong, Shaoxiang Dang, Jing Huang, Yingbing Chen, Yifan Chang, Tianyi Zhang, Shiyuan Deng, Haozhi Wang, Yangkai Wei, Wenqian Li, Han Yang, Kaiwen Zhou, Huaping Liu, James Cheng, Rui Shao, Donglin Wang, Yaochu Jin, Jianye Hao, Ying-Cong Chen, Yinchuan Li（* 通讯：Ying-Cong Chen, Yinchuan Li）
- **机构：** Knowin AI；香港科技大学（广州）；香港中文大学；同济大学；清华大学；哈尔滨工业大学（深圳）；西湖大学；天津大学 等（见项目页）
- **arXiv：** <https://arxiv.org/abs/2609.36012>
- **PDF：** <https://arxiv.org/pdf/2609.36012>
- **项目页：** <https://jethrojames.github.io/awesome-robots-icl/>
- **代码 / 文献库：** <https://github.com/JethroJames/awesome-robots-icl>
- **页数：** 100 pages, 26 figures, 25 tables（arXiv 备注）
- **开源：** **已开源（文献与策展仓库）** — GitHub 提供论文列表与 benchmark 索引；非可运行训练/部署栈
- **入库日期：** 2026-09-30
- **说明：** 与 RA-L 论文 PADP（DOI:10.1109/lra.2026.3734869）**无关**；用户指定 arXiv 链接仅对应本篇综述。

## 核心论文摘录

### 1) 四类「上下文→执行」接口 taxonomy

- 综述按连接上下文证据与执行的接口分四族：**context-conditioned policies**、**geometric demonstration transfer**、**world-model-based control**、**skill- and agent-based execution**；对比各族的迁移假设及训练、对应关系、记忆的角色。
- **对 wiki 的映射：** [../../wiki/entities/paper-in-context-learning-robots-survey.md](../../wiki/entities/paper-in-context-learning-robots-survey.md)、[../../wiki/concepts/robot-in-context-learning.md](../../wiki/concepts/robot-in-context-learning.md)

### 2) 三类学习问题：Acquire / Transfer / Retain

- 从上下文学什么：**新任务**（示范教什么）、**新情境**（换物体/场景是否仍成立）、**下一任务**（经验是否提升后续学习）；并链到评测应区分 **responsiveness to teaching、physical transfer、retained experience**。
- **对 wiki 的映射：** [../../wiki/entities/paper-in-context-learning-robots-survey.md](../../wiki/entities/paper-in-context-learning-robots-survey.md)

### 3) 部署期权重固定与物理递归自改进议程

- 强调 ICL 在 **部署时神经参数固定**，用示范与交互引导已有能力；议程连接组合式任务获取、忠实迁移与 **physical recursive self-improvement**（经验改善后续任务学习能力）。
- **对 wiki 的映射：** [../../wiki/roadmap/depth-icl.md](../../roadmap/depth-icl.md)

## 当前提炼状态

- [x] sources 归档
- [x] wiki 实体页
- [x] 概念页交叉引用
