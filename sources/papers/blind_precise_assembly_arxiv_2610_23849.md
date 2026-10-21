# Blind Precise Assembly — 论文来源归档

> 来源归档（ingest）

## 元数据

- **题目：** Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts
- **作者：** Antonia Bronars、Younghyo Park、Kushal Kedia、Pulkit Agrawal
- **机构：** Massachusetts Institute of Technology（麻省理工学院）
- **预印本：** <https://arxiv.org/abs/2610.23849>
- **项目页：** <https://younghyopark.me/blind-proprioceptive-assembly/>
- **官方代码：** <https://github.com/younghyopark/mjfit>
- **版本：** arXiv v1，2026-10-16
- **入库日期：** 2026-10-20
- **一句话说明：** 研究仅凭关节位置、速度和动作历史进行精密装配，并发布用于 MuJoCo 的程序化精确装配几何生成器。

## 核心论文摘录

### 问题与方法

论文研究在零件与夹具位姿未知、抓取显著随机的条件下，仅用短时关节位置、关节速度和先前动作完成插销入孔、齿轮啮合、螺母旋入。策略不使用视觉、力/力矩或触觉输入；在多种零件、抓取和接触条件的仿真中用强化学习训练。作者发布 `mjfit`，用精确凸体拼合方式程序化构造带真实间隙的 peg-hole、螺纹和齿轮模型，供 MuJoCo 仿真使用。

### 评测与结果

- 插销策略零样本迁移到真实机器人，对多种形状和尺寸的插销达到 73% 成功率。
- 与匹配视觉策略的仿真对照中，本体感知策略对未见零件形状和尺寸的泛化更强，但执行可能更慢。
- 项目页说明公开的是 `mjfit` 几何生成工具；本次核查未发现策略训练/部署代码链接，不能据此认定完整策略实现已经发布。

## 对 wiki 的映射

- [Blind Precise Assembly 实体页](../../wiki/entities/paper-blind-proprioceptive-assembly.md)
- [官方项目页归档](../sites/blind-proprioceptive-assembly.md)
- [mjfit 仓库归档](../repos/mjfit.md)

## 一手来源

- [论文（arXiv:2610.23849）](https://arxiv.org/abs/2610.23849)
- [项目页](https://younghyopark.me/blind-proprioceptive-assembly/)
- [mjfit 官方代码仓库](https://github.com/younghyopark/mjfit)
