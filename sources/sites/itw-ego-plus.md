# ITW-Ego Plus 官方资料归档

- **类型：** site / dataset / egocentric / motion capture
- **项目页：** <https://modalitynet.com/>（ITW+）
- **官方公告：** <https://www.globenewswire.com/fr/news-release/2026/11/05/3396456/0/en/noitom-robotics-introduces-itw-ego-plus-at-corl-2026-in-the-wild-egocentric-data-with-full-body-motion.html>（Noitom Robotics，2026-11-05）
- **官方技术文章：** <https://huggingface.co/blog/noitomrobotics/art-of-modalities>
- **核查日期：** 2026-11-06
- **核对范围：** 官方公告、ModalityNet 公开首页及官方技术文章；数据目录登录后内容和具体样本未核实。
- **通读：** 未完成

## 资料要点

ITW-Ego Plus 在真实环境中结合第一视角视频、全身运动与其他模态；Sol 和 Jupiter 是 Noitom 自研采集设备。官方称 Jupiter 使用少量穿戴设备采集全身运动，不需要动捕服。公告介绍 Body、Wrist、Exo、Touch、Sound 等扩展模态，但不能推断每条记录均包含全部模态。官方技术文章描述更广义 ITW 数据的稀疏身体感知、RGB-D、音频和可选侧视视频，以及手腕手部运动与部分场景的重建；不应把该流程视为每个 Plus 样本的保证。

## 获取与边界

ModalityNet 首页要求注册或登录以浏览数据集；截至核查日，未确认 ITW-Ego Plus 的完整下载、规模、标注规范、许可或可运行源码。HiPHI 开放数据属于另一资源，许可不能直接套用。官方公告称计划在 CoRL 2026 奥斯汀 1105 展位展示。

## 对 wiki 的映射

- [ITW-Ego Plus 实体](../../wiki/entities/itw-ego-plus.md)
