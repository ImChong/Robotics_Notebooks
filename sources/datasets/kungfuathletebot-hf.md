# KungfuAthleteBot Hugging Face 数据集

> 来源归档（dataset；核对日期：2026-10-06）

- **数据集：** <https://huggingface.co/datasets/LuluCao/KungfuAthleteBot>
- **项目页：** <https://kungfuathletebot.github.io/>
- **代码：** <https://github.com/NPCLEI/KungFuAthleteBot>
- **论文：** <https://arxiv.org/abs/2610.03388>
- **Hugging Face 卡片 license 字段：** Apache-2.0（这是数据卡当前元数据；论文则称代码、数据、checkpoint 将在接收后 MIT 发布，授权口径有差异，须按资产条款核验）。
- **资源形态：** GVHMR 人体重建结果、G1 29-DoF 的 robot qpos 及相关处理/示例文件；原始运动员视频未公开。

## 规模与格式

- 卡片当前统计 992 个样本：Ground 822，Jump 170；197 个公开视频经时序切分形成 1,726 个子片段，再经 GVHMR、GMR、人工筛查和后处理。
- qpos 格式示例：30 fps；每帧含根位置 3 维、四元数 4 维与 G1 29 DoF，共 36 个值。
- 类别分布以卡片为准：日常训练 624、剑/刀 111、拳术 98、棍术 90、技巧动作 69。
- Jump 数据仍有源视频噪声；仓库明确提醒，高动态样本未经仿真验证不得直接用于真机训练或部署。

## 版本注意

项目主页和 GitHub README 上部仍显示旧版 848 样本说明，而 Hugging Face 数据卡及 GitHub README 后续统计显示 992。新论文附录 C 也给出 Ground 822 + Jump 170，但附录 E 对释放数据仍描述为 848 筛选样本。因此本归档分别记录网页当前口径与论文原文口径，不将两者混成同一版本。原始视频只获科研/学术用途授权且含可识别影像，数据卡不提供这些视频。

## 互链

- [论文来源](../papers/kungfuathletebot_arxiv_2610_03388.md)
- [项目页来源](../sites/kungfuathletebot.md)
- [代码仓库来源](../repos/kungfuathletebot.md)
- [Wiki 论文实体](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)
