# www-ye.github.io/ego2robot_blog（Ego2Robot 项目页）

- **标题：** Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data
- **类型：** site / project-page
- **URL：** <https://www-ye.github.io/ego2robot_blog/>
- **配套论文：** [arXiv:2608.02580](https://arxiv.org/abs/2608.02580) — [`sources/papers/ego2robot_arxiv_2608_02580.md`](../papers/ego2robot_arxiv_2608_02580.md)
- **代码：** <https://github.com/www-Ye/ego2robot_blog>（2026-11-12 已开放合成流程及 RoboTwin 评测代码）
- **配套源码归档：** [代码入口、流程与依赖](../repos/ego2robot-ye-code.md)
- **入库日期：** 2026-08-15

## 一句话摘要

官方博客：1,940 h 第一人称视频 → 15 形态 18,561 h 合成机器人数据；扩展 RoboTwin 2.0 四轴解耦评测；ARX ACone 真机 few-shot。

## 开源核查（2026-08-15）

历史核查：当时仓内只有 `index.html` / 图 / 视频，尚无管线代码。这条记录**仅代表 2026-08-15 的快照，不能用于当前开源状态**。

## 开源核查更新（2026-11-12）

作者官网 `Code` 跳转到 `https://github.com/www-Ye/ego2robot_blog`。现仓库根目录存在 `ego2robot/`、`configs/`、`scripts/run_pipeline.py`、`scripts/convert_to_lerobot.py`、`evaluation/robotwin/` 与 README（并非纯静态网站）。可读取并使用数据合成管线和评测相关实现；依赖其他模型仓库、权重/视频输入/本体与相机标定。**未核实论文全部合成数据与预训练模型 checkpoint 的完整公开下载**，不可将“代码已开源”等同于完整一键复现。

代码细节见 [Ego2Robot 源码运行链归档](../repos/ego2robot-ye-code.md)。

## 关联资料

- 论文归档：[`sources/papers/ego2robot_arxiv_2608_02580.md`](../papers/ego2robot_arxiv_2608_02580.md)
- wiki：[`wiki/entities/paper-ego2robot.md`](../../wiki/entities/paper-ego2robot.md)
