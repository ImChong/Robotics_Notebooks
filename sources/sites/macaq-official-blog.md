# MACaQ 官方交互介绍（Huawei CSL）

- **类型：** site / 官方博客交互演示
- **链接：** https://huawei-csl.com/blog/macaq/
- **论文：** https://arxiv.org/abs/2611.10649v2
- **代码：** https://github.com/huawei-csl/MACaQ
- **模型集合：** https://huggingface.co/collections/huawei-csl/macaq
- **作者机构：** 华为计算系统实验室（Huawei Computing Systems Lab，瑞士）
- **入库日期：** 2026-11-16
- **核对范围：** Overview、Performance、Examples、Integrations、Method 段，vLLM / llama.cpp 命令与官方代码开放说明。
- **通读：** 已完成
- **沉淀到 wiki：** [MACaQ](../../wiki/entities/paper-macaq.md)

## 核心内容

作者演示低比特量化在文本、图像、视频、音频、3D 重建中沿用逐线性层变换；说明融合 MXFP4 路径上增加在线缩放和块旋转的代价，以及 vLLM、llama.cpp、Diffusers 的实际使用入口。演示中的可选模型和交互指标可能随站点更新，**不同工作负载、模型大小与后端的加速倍数不可混为同一实验**。

## 开源状态与边界

**已开源**：页面直链官方 GitHub，README 标明 Apache-2.0；项目不同后端支持的位宽与模型不同。官方介绍不是机器人闭环成功率实验。
