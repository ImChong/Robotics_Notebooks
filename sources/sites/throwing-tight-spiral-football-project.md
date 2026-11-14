# Tight Spiral：美式橄榄球紧密螺旋投掷（官方项目页归档）

- **项目页：** https://www.zaidmahboob.com/projects/tight-spiral/
- **论文：** https://arxiv.org/html/2608.16642v2
- **视频：** https://youtu.be/gw1AeG-Y614
- **作者：** Zaid Mahboob、Bowen Weng（爱荷华州立大学）
- **核对范围：** 查看作者项目页项目介绍、What a tight spiral is、Why it's hard、How it works、Results、Setup、Cite 及可见外部链接，并与 arXiv v2 核对。
- **通读：** 已完成
- **核对日期：** 2026-11-14

## 开放状态

- **论文：公开。** arXiv:2608.16642v2 已提供 PDF 与 HTML，Humanoids 2026 已接收。
- **视频：公开。** 作者项目页提供 YouTube 演示。
- **项目控制源码、参数配置或运行入口：未见公开项目专用仓库。** 项目页导航仅指向 Paper / Watch on YouTube；页脚的 GitHub 为作者个人资料，不是本研究的可运行实现。不得凭学术论文中的软件工具名推断完整项目开源。
- **原始实验数据/查表控制器：** 页上未列公开下载资源。

## 官网重点核对

1. 真机使用 29-DoF G1、7-DoF Dex3-1、改装拇指甲以及 6 相机 360 Hz OptiTrack。
2. 离线投掷轨迹由 Crocoddyl + Pinocchio 生成，腰/臂模型 10-DoF、0.6 s。
3. 出手阶段 15 × 4 ms MPC 时域；单次原始求解约 500 ms，实际运行改为离线 52 维状态查表。
4. 最高 93.6% 自转效率、最佳 0.286 rad 球头角、最高 5.35 m/s 线速度和 14.5 rad/s 角速度，均非重复性保证。
5. 61 次随动试验平均 78.5 ms；无随动时，平均自转效率 67.1%→60.9%，球头角 18.9°→34.7°。

## 对 wiki 的映射

- [唯一论文节点](../../wiki/entities/paper-throwing-tight-spiral-football.md)
- [论文全文通读归档](../papers/throwing_tight_spiral_football_arxiv_2608_16642_v2.md)

## 参考来源

- https://www.zaidmahboob.com/projects/tight-spiral/
- https://arxiv.org/html/2608.16642v2
