# Online Video Understanding: OVBench and VideoChat-Online｜在线视频理解：OVBench 与 VideoChat-Online

- **类型：** CVPR 2025 paper，pp. 3328–3338
- **作者：** Huang Z, Li X, Li J, et al.
- **原文：** https://arxiv.org/abs/2501.00584
- **CVPR：** https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Online_Video_Understanding_OVBench_and_VideoChat-Online_CVPR_2025_paper.html
- **项目：** https://videochat-online.github.io/
- **官方代码：** https://github.com/MCG-NJU/VideoChat-Online
- **官方权重：** https://huggingface.co/MCG-NJU/VideoChatOnline-4B
- **数据集：** https://huggingface.co/datasets/MCG-NJU/OVBench
- **在线训练数据：** https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT
- **核对范围：** arXiv:2501.00584v2（2025-04-17）正文第 1–6 节与附录/补充材料第 1–5 节（消融、全部基准、QA 生成、具体实例、训练与推理超参数及图注）；官方项目/仓库 README、流式评测和训练入口。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文实体](../../wiki/entities/paper-ovbench-videochat-online.md)

## 原文信息与 Wiki 映射

1. **OVBench：** 时间点嵌入视频流的问答，检查过去记忆、当前感知、未来推断，涵盖六类任务和十六子任务。
   - **对 wiki 的映射：** 强调按提问时刻约束可见帧，避免把离线整段视频输入当成在线能力。
2. **Pyramid Memory Bank（PMB）：** 在不同层次权衡空间分辨率与时间跨度，使用相似度检测及压缩控制记忆量。
   - **对 wiki 的映射：** 主流程图按视觉特征→分层记忆→在线回答。
3. **离线到在线训练：** 视频离线 SFT 后混合在线交错对话训练；提供不同评测协议。
   - **对 wiki 的映射：** 源码入口分别说明 SFT、streaming 和 sliding-window。
4. **结果与开源边界：** 官方 README 给出 VideoChatOnline-4B OVBench 54.9；代码当前公开 ViT 特征记忆实现，README 提示 KV-cache 特征版本尚待后续。
   - **对 wiki 的映射：** 不混淆在线/离线指标，不宣称 KV-cache 完整开源。

- [x] 已核读 arXiv v2 全文与补充材料（HTML 原文含正文、全部表格/图注与附录章节）；论文所述模型精度不等于本次实机复现。

## 正文、附录与代码交叉核验（2026-11-16）

- **在线协议：** §5.1 规定 streaming 在提问时刻前以 2 fps 摄取视频历史；sliding-window 是此前 32 秒、2 fps，两协议不能混表。论文表 2 对同一 4B 模型报告 streaming **54.9**、sliding-window **53.9**，不应把后者误记成流式指标。
- **PMB 结构：** §4.1 与补充材料 §1.1/§5 给出空间层 `m_s` 2 帧×256 tokens，主层 `m_main` 2 帧×64、时间层 `m_t` 12 帧×16；总预算 **832 tokens**。发生层满时，选相邻帧中相似度最高的一对，移除其中较早一帧并降采样下写。论文定义的 KV-cache 同步方案与**实际公开代码仅含 ViT 特征版**需要分开陈述。
- **训练消融：** §5.4 表 6 为 offline-only **45.23**、progressive+online **52.42**、interleaved offline+online **51.84**、progressive+interleaved **53.89**，是**不同于表 2 流式 PMB 54.9 的消融设置**；在线 SFT 样本 **96K**。补充材料 §3.2 的在线任务消融中，移除时空动作检测下降 **3.3** 分、移除时间定位下降 **2.7** 分。
- **效率与条件：** 补充材料表 7 在该文 1000 帧比较条件下报告模型约 **8.71 GB VRAM**；这是特定配置与比较基准，不等于任意部署的峰值显存。
- **数据口径：** 正文引言称来自 **7 个数据集**，§3.2 称筛选 **8 个数据集**；原文表述不一致，不能不经核对强行合并。补充材料 §4.1 报告 **1,463** 个视频，§4.2 给出 QA 生成规则，§4.3 给出带时间戳的实例；第 5 节列出训练及推理超参数。
- **可复现边界：** 已核对官方入口及代码归档；未下载权重/数据进行完整训练或模型推理，因此精度和延迟只归于原论文报告。
