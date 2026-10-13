# VICAR 官方项目页归档

> 来源归档（官网 / 项目展示）

- **项目名称：** VICAR: Video Imitation with Contact-Aware Retargeting for Humanoid Interactive Skills
- **网址：** <https://ember-center-berkeley.github.io/VICAR/>
- **所属团队：** EMBER Center @ UC Berkeley
- **论文：** <https://arxiv.org/abs/2610.13079>
- **代码与演示仓库：** <https://github.com/ember-center-berkeley/VICAR>
- **入库日期：** 2026-10-13
- **关联 wiki：** [VICAR 实体](../../wiki/entities/paper-vicar-contact-aware-video-imitation.md)

## 页面内容

项目页呈现从第三人称人体演示到人形机器人技能的总体流程：

1. 恢复人体动作和任务相关场景几何；
2. 用接触、运动速度与碰撞间隙约束完成重定向；
3. 通过可移动接触点生成/增强动作；
4. 由人形跟踪策略执行参考动作。

项目页展示 10 类技能的视频和可交互的 Viser 动作浏览器，包括六种发球、三类拾取/放置动作与爬梯。项目页主要是研究介绍和展示入口；方法、定量评测及真机结果以论文为准，具体代码范围见 [官方仓库归档](../repos/vicar.md)。

## 页面与论文边界

- 页面按论文叙述展示视频、方法图和接触点交互演示；演示本身不代表所有视频均为真机部署（梯子是仿真展示）。
- 运行时全局球/物体位姿依赖 motion capture；不是只输入普通视频即可无传感器部署。
- 页面 GitHub 的内容与外部数据资产许可需分别核查；论文标注 CC BY 4.0 不等于网页内所有模型、视频、轨迹都采用 CC BY。
