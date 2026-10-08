# FreeSpeed 项目页

- **类型：** project site
- **URL：** <https://yuxuanhu9.github.io/FreeSpeed/>
- **核查日期：** 2026-10-08
- **关联论文：** [arXiv:2610.05734](https://arxiv.org/abs/2610.05734)
- **作者与机构：** Yuxuan Hu 等；MARS Lab, Nanyang Technological University；ROKAE Robotics
- **项目实体：** [wiki/entities/paper-freespeed.md](../../wiki/entities/paper-freespeed.md)

## 项目页要点

FreeSpeed 对冻结的动作块策略进行推理时速度控制：按请求速率重采样动作块，再以相邻平移增量的方向不一致度控制动作步长缩放；平移和旋转被缩放，夹爪指令不变。项目页将方向变化较大的动作段关联到抓取和放置等关键阶段。

项目页报告：三类策略、50 个仿真任务上，保持逐任务 1× 成功率的实际执行速率为 0.22×–2.53×；四项真机任务在六种非参考命令下平均成功率 94.0%，冻结策略参考为 93.8%。

## 资源开放状态

核查时项目页公开了方法、实验结果和真机视频，并显示 **“Code soon”**。页面没有链接可运行的源码仓库或数据下载入口。因此当前按“论文和项目页已公开、官方代码待发布”记录；不提供代码运行步骤，也不推定许可证。