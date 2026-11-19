# CloudRobo-WM-1.0-RoboTwin-20261111｜模型资产版本说明

- **类型：** Hugging Face 上的模型资产标识（非独立论文）
- **模型：** https://huggingface.co/CloudRobo/CloudRobo-WM-1.0-RoboTwin-20261111
- **后训练权重：** https://huggingface.co/CloudRobo/CloudRobo-WM-1.0-RoboTwin-20261111/blob/main/robotwin-posttrain/cloudrobo_wm_bf16.safetensors
- **初始化权重：** https://huggingface.co/CloudRobo/CloudRobo-WM-1.0-RoboTwin-20261111/blob/main/pretrain/initialization_bf16.safetensors
- **部署与校验清单：** https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/CloudRobo_WM/README.md
- **核对范围：** 完整阅读 XPolicyLab 的模型资产下载与 SHA256 说明，并核对公开 HF 文件链接；当前工具未能直接读取 Hugging Face 模型卡全文，未核验全部权重文件内容或许可证。
- **通读：** 未完成
- **入库日期：** 2026-11-19

## 身份不能混用

- 名称中的 1.0 和 20261111 为公开 checkpoint 标识，不能擅自改标 WM1.2。
- 推理所用评测 checkpoint 为 robotwin-posttrain/cloudrobo_wm_bf16.safetensors；XPolicyLab 报告 SHA256: 660bc6bd55d526366ea0f4abccb660d1b5aa1a29c8af1c7bfb1566a4b9e0462d。
- pretrain/initialization_bf16.safetensors 是后训练初始化，不是已评测权重；XPolicyLab 报告大小 14,133,497,112 bytes、SHA256: 9e105d2f047dfa482dfd172ea912ffbcedd8ae5de35143ffe91c502076160a59。
- 华为云文档提到 CloudRobo-WM-7B-Preview 为 RoboTwin 2.0 Clean 数据的后训练预览，但目前未核实其与上述 HF checkpoint 的二进制一致性。

## 工程注意

先下载与校验模型权重及第三方依赖，再使用 XPolicyLab CloudRobo_WM adapter；不能以 HF 文件名证明 WM1.2 专属权重已经发布。

## 关联

- [XPolicyLab 代码与评测](../repos/cloudrobo_wm_xpolicylab.md)
- [华为云 WM1.2 新闻](../sites/cloudrobo-wm12-2026.md)
- [家族 Wiki 实体](../../wiki/entities/cloudrobo-wm.md)
