# HandRSI 模型与 Harness：Hugging Face 发布核对

- **模型卡：** https://huggingface.co/HandRSI/HandRSI
- **论文：** https://arxiv.org/abs/2611.22164 （[归档](../papers/handrsi_arxiv_2611_22164.md)）
- **代码：** https://github.com/GinKenneth/Hand_RSI （[归档](../repos/handrsi.md)）
- **核对日期：** 2026-11-20
- **核对范围：** 模型卡的 checkpoints、复现使用方式、测试表、checkpoint 组成、许可、sha256 说明；未下载或本地校验权重文件。
- **通读：** 已完成

## 文件身份

| 发布目录 | 说明 |
|---|---|
| `theta1/` | 第 1 轮更新 ACE-Ego-Hand K-given 得到的 LoRA 适配参数与 heads |
| `theta2/` | 第 2 轮网络；之后三轮无更优权重通过验收，最终 θ3 = θ4 = θ5 = θ2 |
| `harness/H1` … `harness/H5` | 五轮可执行模块链；最终 H5* 含 39 个模块 |
| **最终系统** | **θ2 + H5***；与三种速度分层不可混为一个配置 |

- **模型卡给出的加载示例：** 先按 ACE-Ego-Hand README 准备其依赖，再用 `python infer_video.py --video clip.mp4 --intrinsics fx,fy,cx,cy --ckpt <path>/theta2 --harness <path>/harness/H5 --out results/clip`。
- **许可：** 训练权重 **CC BY-NC 4.0**；Harness 代码 **MIT**；MANO 不随权重发布且使用单独许可。模型仓未提供一键云推理服务。
- **Wiki：** [HandRSI](../../wiki/entities/paper-handrsi-recursive-self-improvement-hand-pose.md)
