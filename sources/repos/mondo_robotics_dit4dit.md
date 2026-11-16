# Mondo-Robotics/DiT4DiT

> 来源归档（ingest）

- **标题：** DiT4DiT — Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- **类型：** repo
- **官方入口：** <https://github.com/Mondo-Robotics/DiT4DiT>
- **论文：** <https://arxiv.org/abs/2603.10448>
- **项目页：** <https://dit4dit.github.io/>
- **机构：** Mondo Robotics；HKUST(GZ)；HKUST
- **许可证：** MIT（以仓库 LICENSE 为准）
- **入库日期：** 2026-06-10
- **一句话说明：** DiT4DiT 官方代码：双 DiT Video-Action Model 训练、评测与部署；Cosmos-Predict2.5 视频骨干 + flow-matching 动作头联合优化；含 LIBERO / RoboCasa / Unitree G1 真机管线。

## 仓库要点（公开 README 索引）

| 模块 | 说明 |
|------|------|
| **训练** | Video DiT + Action DiT 联合 dual flow-matching；三时间步 $\tau_v,\tau_f,\tau_a$ |
| **推理** | 可仅动作预测，或同步生成未来视频计划 |
| **部署** | WebSocket policy server；G1 单目 egocentric 闭环 |
| **权重** | 项目页 / Hugging Face 分发（以仓库当前 release 为准） |

## 对 wiki 的映射

- [DiT4DiT 论文实体](../../wiki/entities/paper-dit4dit-video-action-model.md)
- [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) — 同团队后续人形实时 WAM 工作

## 固定版本源码复核（2026-11-13）

- **版本：** [66a6f3a12e2c8740157c1e478795952c040d31dd](https://github.com/Mondo-Robotics/DiT4DiT/tree/66a6f3a12e2c8740157c1e478795952c040d31dd)
- **核对范围：** 根README、docs/libero.md、server_policy.py、framework/DiT4DiT.py全文；LIBERO ModelClient初始化/step/反归一化，Cosmos25.forward与extractor非生成分支、ActionDiT构造/predict_action完整函数。
- **通读：** 已完成

README提供训练/检查点、LIBERO/RoboCasa客户端与G1/WBC入口。LIBERO客户端拼接两视角、state逐维sin/cos与padding，WebSocket传examples；服务器加载BaseFramework，主模型调Cosmos25再调FlowmatchingActionHead返回normalized_actions。客户端按chunk缓存、反归一化和夹爪阈值处理后执行。

无future输入且不请求预测视频时extractor默认只跑首步并hook隐状态，不做完整视频去噪/解码；动作头按checkpoint的num_inference_timesteps做Euler采样。客户端use_ddim参数不应被当作该动作头实际采样实现。静态核读未进行机器人或GPU复现。
