# You don't need long context

> 来源归档（paper）

- **标题：** You don't need long context
- **作者：** Peter Belcak, Saurav Muralidharan, Jindong Jiang, Yonggan Fu, Wonmin Byeon, Hanrong Ye, Ryan Angilly, Ali Taghibakhshi, Shih-Yang Liu, Siru Ouyang, Greg Heinrich, Pavlo Molchanov
- **机构：** 英伟达研究院（NVIDIA Research）
- **类型：** paper / technical report
- **发布日期：** 2026-11-03（arXiv v1，31 页，预印本）
- **链接：** https://arxiv.org/abs/2611.05306
- **HTML：** https://arxiv.org/html/2611.05306v1
- **PDF：** https://arxiv.org/pdf/2611.05306v1
- **DOI：** https://doi.org/10.48550/arXiv.2611.05306
- **官方研究入口：** https://nv-dler.github.io/ 、 https://research.nvidia.com/labs/lpr/lfi/
- **核对范围：** arXiv v1 HTML 正文 §1–6，图表说明及附录 A–G 的压缩策略、成本假设、基准结果、选择性读取示例和提示词结构；官方 DLER 论文列表与 LFI 研究方向。
- **通读：** 已完成
- **代码开放状态：** 2026-11-24 核对官方 DLER 论文条目仅列 arXiv；论文公开了实验框架名称及附录提示词，但未提供本研究 EoCC/SAC 完整实现和实验数据的专属仓库。mini-swe-agent v2.4 是引用的既有基础框架，不能等同于本论文的修改版。

## 研究问题与方法（§1–2，Fig. 3）

长输入是资料本身很长，长程任务是交互历史随动作增长。FHA 保留全部历史；EoCC 达阈值后生成交接状态，在新会话保留任务及状态；SAC 每执行一次环境动作并观察结果即更新语义状态、重置会话。长文档留在文件系统，由 grep、sed、doc-read 等按需读取，避免全部注入模型。总上下文长度是各会话长度之和，峰值是最大单个会话；两者不能混用。

## 实验记录（§3–5，附录 C–F）

- H100 饱和解码实验（Fig. 1、4）：256K 降至 16K，GLM 5.2、Kimi K2.6、Qwen 3 Coder Next 的吞吐分别提高约 3.6、5.6、7 倍；输入前缀已缓存，固定输出 1024 tokens，并非所有端到端请求的加速。
- 六个长程基准（SWE-bench Verified/Multilingual、Terminal-Bench 2.1、FRAMES、tau2-bench、BFCLv3）：EoCC 准确率相对 FHA 变化范围，Opus 为 −3.7 至 +5.6 个百分点、GPT 为 −3.3 至 +4.4。FRAMES 存在下降，不能说压缩永远提升。
- 七个长输入基准（LongBench v2、LOONG、AA-LCR、NIAH、RULER、Counting-Stars、BABILong）通过文件选择性读取与 EoCC 后备机制运行。表 15 的 RULER mean recall 从 62.5%/65.5% 升到 95.7%/96.4%（Opus/GPT，16K/12K）；表 11 LongBench v2 为 68.0→67.6、70.8→72.6。
- 表 2 的 SAC：Opus 在 SWE-bench Verified 70.0→70.2%，Terminal-Bench 47.2→51.8%；GPT 分别 84.4→82.4%、87.6→84.3%。四组峰值上下文中位数 7–9K、最大值 15–37K。最佳 EoCC 在这四组均高于 SAC。
- 附录 D 表 9：量化 Qwen 3 Coder Next 在 SWE-bench Verified 的 16K/12K EoCC 从 53.0% 降至 3.6%，说明短预算需要针对模型校准。

## 成本与局限（§3.1、§5.1–6，附录 A–B/G）

作者根据三个开权重模型的 16K/256K 解码成本比取近似均值 0.20，并用线性插值构造闭权重模型的估计优化价格，维持输出/新输入/缓存输入价格比例。长程与长输入的估算节省不是已经发生的 API 账单；新输入和输出量可能因重置增加。交接状态应保留任务、已验证事实、变更、验证、失败路径和后续动作，而非平均概括全部历史。压缩有损，必要历史应外部持久化、支持回查。SSSM 是作者提出的自然语言状态后训练方向，本文未交付经过训练的独立 SSSM checkpoint。本文未证明所有任务都能使用短上下文，也未评测机器人真机任务。

## 对 wiki 的映射

- [独立论文实体](../../wiki/entities/paper-you-dont-need-long-context.md)
- [官方研究入口归档](../sites/nvidia-you-dont-need-long-context.md)
