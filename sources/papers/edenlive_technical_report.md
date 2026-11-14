# EdenLive 技术报告全文核读

- **原文：** <https://assets.vast-eden.ai/assets/edenlive/EdenLive-tech-report.pdf>
- **版本：** 官方2026-10技术报告，2026-11-13下载；无arXiv编号。
- **核对范围：** §1–§8正文全文与表1–4、图注，作者/致谢浏览；无研究附录。另读官方页及README、固定版本实时模型会话/条件编码路径。
- **通读：** 已完成

引擎执行状态转移、EdenLive只呈现外观。DINOv3-L/16以配对位置对齐+多正样本InfoNCE+域对抗学习共用空间，随后冻结；reference沿通道而非token拼接。先teacher-forcing再自生成rollout/replay蒸馏，推理四步无CFG、首帧sink+滚动窗口及RoPE重定位。

H1005B720p29.80FPS包含条件/解码/CPU传输，排除proxy渲染、文件编码、网络；31.04是更窄生成吞吐。首帧暖启动1.162s、稳态P95块间391.54ms；两块交互延迟另加传输，不可当24FPS=42ms响应。14B480p26.76FPS要8H100，不是单卡。

30s长时评测Eden用动态prompt而对照静态，Reference-DINO沿用自身微调编码器；需结合独立相机/外观指标。AgentGarten的平移/Reference-DINO优于Eden，不能说所有项胜出。45例VLM偏好非人工长期验收。表1DOVER文字称0–100，但表值约9–10，保留原表口径不任意换算。

具身增广只展示视觉结果，无下游机器人成功率或动作标签保持量化；显式引擎状态正确不保证生成像素正确。身份对应、遮挡重访外观记忆、细节控制仍有限。
