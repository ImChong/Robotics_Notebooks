# GigaWorld-1

> 来源归档（国内具身开源全景）

- **标题：** GigaWorld-1
- **类型：** repo
- **机构：** 极佳视界
- **链接：** https://github.com/open-gigaai/giga-world-1
- **分类：** 世界模型
- **入库日期：** 2026-09-06
- **一句话说明：** 极佳视界 开源项目 GigaWorld-1（世界模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/paper-gigaworld-1-policy-evaluation.md`](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/paper-gigaworld-1-policy-evaluation.md](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)

## 固定版本源码核读（2026-11-13）

- **版本：** [0ad5e2b10424f6680f9836fd7055fb43d8e5855d](https://github.com/open-gigaai/giga-world-1/tree/0ad5e2b10424f6680f9836fd7055fb43d8e5855d)
- **核对范围：** 根README全文、scripts/infer/run_infer_nano_i2v.sh、infer/infer_giga_world.py全文；GigaworldFunCtrlPipeline.__call__/stage1_sample完整函数。未核所有训练/闭环策略接入模块。
- **通读：** 已完成
- 开源为部分：Nano/Pro Stage-1权重、SFT/DMD训练入口、离线控制视频生成、toy数据与部分benchmark；蒸馏权重、RL后训练和完整加速框架仍待发布。
- CLI明确use_dmd=False；只减少num_inference_steps不能等同蒸馏模型。README称control可选/t2v纯文本，但stage1_sample断言control_latents非空，当前FunCtrl入口不能保证无control可运行。
- nano_i2v.sh给首帧/控制视频/scene LoRA，加载VAE/T5/Transformer，逐chunk切控制、构造三级history+anchor，stage1_sample调用Transformer并由scheduler更新，decode/累积导出视频。该离线脚本不调用机器人策略，不是完整WMBench闭环。
- Python默认fps24，README写10而launcher显式10；输出fps是视频封装时间基，不是生成吞吐。代码prepare_control_video_latents的9 latent窗口仍有硬编码，应对齐配置。静态核读，未跑GPU。
