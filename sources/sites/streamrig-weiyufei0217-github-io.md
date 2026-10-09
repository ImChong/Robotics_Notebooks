# StreamRig 官方项目页

- **类型：** site / project page / research demo
- **链接：** <https://weiyufei0217.github.io/StreamRig/>
- **论文：** <https://arxiv.org/abs/2609.40244>
- **代码：** <https://github.com/WeiYuFei0217/StreamRig>
- **作者主页：** <https://weiyufei0217.github.io/>
- **作者 / 团队：** Yufei Wei 等；浙江大学、华南理工大学
- **一句话说明：** 项目页展示 StreamRig 的流式多相机里程计架构、四组数据评测、交互式三维轨迹与视频。

## 项目页确认的信息

- 页面将方法概括为联合利用 rig 几何、紧凑因果状态和连续里程计；冻结多视图前端，只训练 74.6M 参数的轻量后端。
- 方法组件包括 frozen multi-view front-end、Rig-Resampler、CausalBridge 和 pose head；通过 periodic re-anchoring 将相对位姿连接为连续轨迹。
- 页面介绍 NCLT（5 相机）、TartanGround（模拟 4 相机）、KITTI-360（4 相机）和 ZJH 真机人形机器人（4 相机）。
- 页面展示交互点云地图、不同相机配置与位姿轨迹对比，以及方法 / 数据集演示视频。
- 页面列出的消融结论包括：mixed windows、displacement-normalized translation 和 per-query anchor snapshot 可降低漂移；移除 group-relocalization 预训练后，平移漂移增加 4.6–14.9 倍。该倍数是项目页所报告的消融结果，不是跨数据集保证。
- 页面报告 5 相机输入每次到达耗时 26.2 ms、显存 2.6 GiB；此测量配置与[作者主页](https://weiyufei0217.github.io/)所述四相机 50 Hz 不是同一设置，不要混用。

## 关联归档

- [StreamRig 论文题录与摘要](../papers/streamrig_arxiv_2609_40244.md)
- [StreamRig 官方仓库与复现入口](../repos/streamrig.md)
- [论文详情](../../wiki/entities/paper-streamrig.md)
- [项目详情](../../wiki/entities/streamrig.md)
