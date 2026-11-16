# ABot-Manipulation

> 来源归档（国内具身开源全景）

- **标题：** ABot-Manipulation
- **类型：** repo
- **机构：** 阿里巴巴
- **链接：** https://github.com/amap-cvlab/ABot-Manipulation
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **一句话说明：** 阿里巴巴 开源项目 ABot-Manipulation（VLA/操作模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/paper-abot-m05-mobile-manipulation-wam.md`](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/paper-abot-m05-mobile-manipulation-wam.md](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md)

## M0.5固定版本源码核读（2026-11-13）

- **版本：** [7642747ed2817b241dde5df06e17ee80192718ad](https://github.com/amap-cvlab/ABot-Manipulation/tree/7642747ed2817b241dde5df06e17ee80192718ad)
- **核对范围：** 根README、GETTING_STARTED.md、wam/server.py、core/server_policy.py全文；pipeline_wam.py初始化/检查点配置、__call__、_run_denoising_phase、prepare_kv_cache完整函数。
- **通读：** 已完成
- 2026-07-21已发布M0.5推理/评测与权重，旧coming soon过时；M0训练分支与M0.5不能混称。
- 该inference-only构建明确只支持lat+act双流MoT，_resolve_transformer_load对enable_m3=True直接报错。故不可将论文三流LAM/DreamForcing训练画作该发布的实际运行链，也不宣称全部训练配方可复现。
- server加载WAM_CONFIGS与WAMServerPolicy，infer分reset/cache prefill/chunk。采样先video再action，末步更新cache；prepare_kv_cache用真实观测/状态清预测并更新历史。返回postprocess动作与可选预测视频。日志logical actions/sec与墙钟actions/sec不同，无统一Hz实测外推。
- README链接检查点和双环境LIBERO/RoboCasa评测；主代码Apache2.0。本次静态阅读，未跑GPU/真机。
