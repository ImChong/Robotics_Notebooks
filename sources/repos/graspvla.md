# GraspVLA

> 来源归档（国内具身开源全景）

- **标题：** GraspVLA
- **类型：** repo
- **机构：** 银河通用
- **链接：** https://github.com/PKU-EPIC/GraspVLA
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **一句话说明：** 银河通用 开源项目 GraspVLA（VLA/操作模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-graspvla.md`](../../wiki/entities/cn-os-graspvla.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-graspvla.md](../../wiki/entities/cn-os-graspvla.md)

## 官方资源补核（2026-10-05 / 2026-11-21）

- **代码：** <https://github.com/PKU-EPIC/GraspVLA>（commit `6acd035`）；项目资源以此规范仓 README 为准。
- **权重：** <https://huggingface.co/vegebirrd/GraspVLA>。
- **数据：** SynGrasp-1B（<https://huggingface.co/datasets/vegebirrd/SynGrasp-1B>），README 2026-08-19 公告已发布；不能沿用早期“待发布”说明。
- **辅助仓库（README 列出）：** 仿真游乐场 `MiYanDoris/GraspVLA-playground`、真机控制接口 `MiYanDoris/GraspVLA-real-world-controller`。
- **入口：** `uv sync --locked`；推理服务 `vla_network.scripts.serve`（ZMQ ROUTER 接收图像/本体数组，经 `VLAAgent` 推理并线性/轴角插值为增量动作）与客户端验证 `vla_network.scripts.offline_test`（支持 mock 或 `.npy` 回放并绘制左右视角预测 bbox）；分布式训练 `torchrun --module vla_network.scripts.train_vla`（支持 `cot_flow_matching` 等 `--pred` 模式、FSDP 与 `dinosiglip` 2D 主干）。checkpoint 同目录需有 `config.json`、`preprocessor.npz`；服务默认还需加载 `internlm/internlm2-1_8b`。
- **机制：** 合成抓取动作与互联网语义共同学习，感知思维链连接语义理解与 flow 动作头。
- **核对范围：** 通读 `PKU-EPIC/GraspVLA@6acd035` 的 `README.md`、`vla_network/scripts/serve.py` 与 `vla_network/scripts/offline_test.py` 源码及仓库目录结构；未在本机启动 GPU 推理服务或下载百亿级 SynGrasp-1B 数据集。
- **通读：** 已完成
