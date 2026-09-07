# MOSS Transcribe Diarize（Hugging Face Space）

- **类型：** 在线演示 / Gradio Space
- **收录日期：** 2026-09-07
- **Space：** <https://huggingface.co/spaces/OpenMOSS-Team/MOSS-transcribe-diarize>
- **维护方：** OpenMOSS-Team
- **同源仓库：** <https://github.com/OpenMOSS/MOSS-Transcribe-Diarize>
- **权重：** <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>

## 一句话

浏览器上传 **最长约 30 分钟** 的音视频，由远端服务完成 **转写 + 说话人分离 + 时间戳**，返回可读分段结果——零本地 GPU 的快速体验入口。

## 为什么值得保留

- 与 mosi.cn Demo、GitHub Web App 构成 **三层体验**：HF Space（轻量远程）、官方 Demo 站、本地可部署 UI。
- Space 描述明确任务边界（转写 / diarization / timestamps），便于 wiki 写「部署读法」时区分 **演示上限 30 分钟** vs 模型 **128k/90 分钟** 能力。

## 站点摘录（2026-09-07）

来源：<https://huggingface.co/spaces/OpenMOSS-Team/MOSS-transcribe-diarize>

- **输入：** 音频或视频文件（Space 说明上限 **30 minutes**）。
- **输出：** 带说话人标识与时间戳的清晰转写文本。
- **后端：** 调用远程推理服务（非纯浏览器端模型）。

## 对 wiki 的映射

- 主沉淀：[MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md)
- 项目页：[moss-transcribe-diarize-mosi.md](./moss-transcribe-diarize-mosi.md)
- 代码：[moss-transcribe-diarize.md](../repos/moss-transcribe-diarize.md)
