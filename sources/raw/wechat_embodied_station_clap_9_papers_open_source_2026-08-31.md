---
title: CLAP代码模型全开源！9篇开源论文串起跨本体世界模型与VLA
author: 具身智能小站
date: "2026-08-31 09:00:00"
source: "https://mp.weixin.qq.com/s/J62q2IVvvBDyT_8OTR9KZQ"
---

# CLAP代码模型全开源！9篇开源论文串起跨本体世界模型与VLA

📅 2026年8月31日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 9 篇近期机器人与具身智能论文，覆盖人—物交互三维重建、跨本体视频世界模型、流式 VLA、场景重排规划、世界动作模型、VLA 安全攻击、视触觉操作、空间参照系理解与社交机器人编排。它们共同指向一个变化：具身系统正在从单一动作预测，走向可模拟、可流式执行、可诊断并可跨本体迁移的完整闭环。

**综述主线：**本期主线可以概括为三层：CLAP 与 Riemann-1.0 扩展世界模型的数据和本体边界，FlashVLA 将动作生成改造成稳定的流式过程，TrapVLA 与 ESRP 则暴露安全和长时程规划中的新难题；MILO、ViTaR、AlloEgo-VLM 和 MistyPilot 分别从三维交互、触觉校正、空间语义与智能体编排补齐感知—执行接口。

01 · arXiv:2608.27407

🔬 **Reconstructing Humans and Objects in Interaction using Large Reconstruction Models**

📌 **3D Human-Object Interaction · Large Reconstruction Model · Embodied AI · Single-Image Reconstruction**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguawjPCQIAbhf9DbMJmtxEjTN6nu2FTk79qqIiba6eP3WbG701iatmPwkNDE82ESKZB5zc4ibOkCmcm5ZP2hmial2eWaL7Pln1dwMAs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 借助大型重建模型提供的**几何脚手架**，从单张图像恢复细致的人—物三维交互。

📖 单图三维人—物交互重建受到深度歧义、遮挡和物体形状差异影响，既有方法多依赖二维重投影、接触约束以及人体和物体模板拟合。MILO 改用大型重建模型生成保留人—物相对布局与邻近关系的三维网格，再将其分割为人体和物体部分：人体侧拟合参数化身体模型，若有模板则进一步对齐物体。该流程把复杂优化转化为对 LRM 网格的结构化解释，在多个基准和交互场景中取得优于既有方法的重建精度，论文同时开放代码。

💡 大型重建模型的价值不仅是生成几何，更在于提供**交互关系的初始结构**。

🔗 项目链接： 项目页与代码

🔗 资料来源： https://arxiv.org/pdf/2608.27407

02 · arXiv:2608.27406

🔬 **CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators**

📌 **Video World Model · Cross-Embodiment · Zero-Shot Simulation · Robot Learning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZxbpPc74S21VwQIkRCrUNdFELiafThfcc6Fm5aKybria2lO7CwqvJ0hImpznemSz2jVqX7OJAG7mSpqNGjLjZIbib8tfPpU6mIjI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 统一末端位姿、语言与潜动作，让视频世界模型学习**跨本体的通用物理先验**。

📖 现有动作条件视频模型通常绑定单一机器人形态，难以利用包含人类和多类机器人的异构视频。CLAP 以末端执行器位姿、自然语言和学习式潜动作协调不同本体的动作空间，并采用课程式跨本体训练：先从无标注视频学习基础物理先验，再落地到末端动作空间进行真实任务零样本部署。模型在 DROID 等环境中接近或超过单本体先进模型，少样本适配收益进一步扩大；通过跨策略规划和基于世界模型的强化学习微调，还能提升 π0.5 与 MolmoAct-2 等机器人策略。项目明确开放**全部代码与模型**。

💡 若物理规律能够跨本体共享，视频世界模型就可能成为**通用仿真底座**。

🔗 项目链接： 项目页GitHub：代码与模型

🔗 资料来源： https://arxiv.org/pdf/2608.27406

03 · arXiv:2608.27384

🔬 **FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference**

📌 **Vision-Language-Action · Streaming Decoding · Asynchronous Control · Flow Matching**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZia8o6VGE3cb3iaKo8DibibW8N69E5aEDA2Xhb8TshqEeDUAPsNMOzUUJjqwcWsB3Qiciax3wsfERj7A7pu9qcEAIpqjn6icpXkCOd1I/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 用多噪声动作缓存实现**每步输出一个可执行动作块**，兼顾低延迟与异步连续性。

📖 基于流匹配的 VLA 需要多轮迭代解码，真实部署同时受到推理延迟和异步执行不稳定的限制。FlashVLA 维护包含不同噪声级别动作块的流式缓存，并以块级因果注意力逐步解码，使每次推理都能产出一个可执行动作块。块级自回归结构还隐式保持动作连续性，无需额外预测未来状态。仿真和真机实验表明，该框架在维持较强任务性能的同时显著提高推理速度，并可在单张 GPU 上实现 **不低于 30Hz** 的控制频率与平滑异步执行。

💡 VLA 的异步控制不应只是并行线程问题，而应成为**解码过程本身的结构**。

🔗 项目链接： https://github.com/z-lab/flashvla.git

🔗 资料来源： https://arxiv.org/pdf/2608.27384

04 · arXiv:2608.27371

🔬 **Embodied Scene Rearrangement Planning**

📌 **Embodied AI · Scene Rearrangement · Long-Horizon Planning · Benchmark**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguasZlFiaEZQ2zh2xSEdtubWU7o4yFuMgClw08YnMQbd4K2cGkiaIwoxicuF6rtHzDu9aiahbs63tnWIcEYeFa38icicIQaG7LSXH95zA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 仅凭第一视角观察与俯视目标布局，规划机器人完成**受遮挡的三维家具重排**。

📖 ESRP 要求具身智能体把三维场景中的家具重新布置为目标构型，但只能使用第一视角观察和俯视目标布局，既不能访问全局状态，还必须面对物体相互遮挡。为研究局部观察与全局布局对齐下的长时程规划，作者构建基于 OmniGibson 的 ESRP-Bench，包含 **5400 余组场景和 8200 个物体**，并设计三级评测指标。论文提供分层任务—运动规划、视觉语言模型、模仿学习和强化学习四类基线；实验显示现有方法仍难以高效完成重排。

💡 场景重排的难点不是识别单件家具，而是持续维护**局部观察与全局目标的对应关系**。

🔗 项目链接： https://pie-lab.cn/ESRP/

🔗 资料来源： https://arxiv.org/pdf/2608.27371

05 · arXiv:2608.27033

🔬 **Riemann-1.0: An Embodied World Action Model for Physical AI**

📌 **World-Action Model · Physical AI · Autoregressive Model · Embodied Pre-training**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZFnA8NHB3UaWutFWdFbt8RdYF4qibPgpFOzdfwzPt8uTlXB0PtvH9kQ5EJHFibAAoxP4uzSgZSV5bewbXMJxe9lgxstLSPLqaHg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 在统一因果序列中联合建模多视角、机器人状态与动作，兼任**策略和世界模拟器**。

📖 Riemann-1.0 是全因果自回归世界动作模型，把多视角视觉、机器人状态和本体特定动作统一为因果状态转移，使同一模型既能在线执行策略，也能进行动作条件视觉世界模拟。渐进式具身预训练进一步在统一目标下吸收第一视角人类视频、手持夹爪示范和异构机器人轨迹，训练数据超过 **20 万小时**。模型在 RoboTwin2.0、LIBERO 和 RoboCasa-365 上分别达到 **94.3%、99.0% 和 62.6%**；长时程真机任务成功率为 85.0%，比最强开源基线高 15 个百分点。

💡 把策略学习与世界模拟合并后，规模化的对象将从动作数据扩展为**完整具身经验**。

🔗 项目链接： https://riemann-dynamics.github.io/Riemann-1.0-Website

🔗 资料来源： https://arxiv.org/pdf/2608.27033

06 · arXiv:2608.26578

🔬 **TrapVLA: Trapping Vision-Language-Action Models in Configured Failure Modes**

📌 **VLA Security · Backdoor Attack · Robot Safety · Adversarial Learning**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZqhSPC53Ix9fBEY8oRa7kEGAo47EKHxibcJlNkzodExUqlYjseTgfPjgaBickgrvaqUtFY5CAknZn1JoolBXJLPSt5n4ZribQf4o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 不再满足于让任务随机失败，而是通过隐蔽文本触发器控制机器人进入**指定失败模式**。

📖 传统 VLA 后门攻击通常把任何任务失败都视为攻击成功，却无法控制机器人具体怎样失败。Configured Failure Trapping 要求隐蔽文本触发器诱导预设错误，例如让抓取位置产生指定偏移。为此，论文构建目标轨迹合成引擎和自动化失败忠实度评测套件，并发布覆盖四类失败模式的 Trap-LIBERO 与 Trap-RoboTwin。TrapVLA 将稀疏动作偏移视为关键挑战，显式学习触发器诱导的动作残差。仿真与真机实验显示，它能稳定注入配置失败，同时大体保留干净数据上的正常性能。

💡 VLA 安全评测需要从“会不会失败”推进到**攻击者能否控制失败方式**。

🔗 项目链接： https://john-liua.github.io/TrapVLA/

🔗 资料来源： https://arxiv.org/pdf/2608.26578

07 · arXiv:2608.15816

🔬 **ViTaR: Visuo-Tactile Residual Adaptation for Foundation VLA Manipulation**

📌 **Visuo-Tactile Learning · Vision-Language-Action · Residual Adaptation · Contact-Rich Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYMHzLfjTyvsXegibaNeibHF17tx0ibJibooRPmLeZHtZsgqGk1VJBRYfpFgYialreOR6vibDG3smubEicWRZzsWnMjjycR3YnlQmdK7w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 冻结基础 VLA，只让触觉选择并缩放**有界残差修正**，避免触觉覆盖视觉语义先验。

📖 基础 VLA 具有广泛视觉语义先验，却无法区分接触是否建立、滑移或丢失；直接把触觉融入模型可能导致遗忘，在线强化学习又需要高风险探索。ViTaR 将触觉从动作生成输入改为执行调制器，在冻结 VLA 之上选择并缩放有界残差：Effect-Guided Modeling 判断局部修正是否合理及其类型，Residual Action Modulation 再依据实时视触觉观察连续调节增益。在覆盖 7 个接触密集任务的 UniVTAC 上，平均成功率达到 **61.3%**，比冻结基础模型提高 **30.6 个百分点**，并可迁移到真机传感器噪声和动力学。

💡 接触反馈不必重写策略方向，更适合负责**校准动作执行**。

🔗 项目链接： https://icr-lab.github.io/ViTaR

🔗 资料来源： https://arxiv.org/pdf/2608.15816

08 · arXiv:2608.15605

🔬 **AlloEgo-VLM: Disambiguating Allocentric and Egocentric Reference Frames in Vision-Language Models**

📌 **Vision-Language Model · Spatial Reasoning · Reference Frame · Embodied AI**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgua8MoHSeBVVPJsglvTJOaFdyuBT6EpJIPpdd6Le5ibmYMN7VfS4uI5k5EaLPdyic3cQhj3NUuSSUASNMH8SqSHAIOXJoyatjsw1w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

✨ 显式区分**自我中心与环境中心参照系**，减少空间指令省略视角时的语义歧义。

📖 空间语言经常省略参照系，同一句方向描述可能来自观察者视角，也可能来自对象或环境视角，导致 VLM 在具身任务中给出不一致答案。AlloEgo-View 数据集用“图像—问题—视角特定答案”三元组描述场景、参照物、目标物、朝向、参照系和视角类型。在此基础上，AlloEgo-VLM 通过监督微调集成到现有 VLM，即使问题含糊也能区分 allocentric 与 egocentric 语义。作者还在 NVIDIA Isaac Sim 的开放式物体搜索任务中部署验证；实验揭示现有 VLM 的参照系短板，并显示该方法具有较强消歧能力。

💡 机器人理解“左与右”之前，必须先回答**站在谁的视角看**。

🔗 项目链接： https://github.com/CKL9001/AlloEgo-VLM.git

🔗 资料来源： https://arxiv.org/pdf/2608.15605

09 · arXiv:2608.15549

🔬 **MistyPilot: Enabling Social-Robot Control through Multi-Agent LLM Skill Orchestration**

📌 **Social Robot · Multi-Agent LLM · Skill Orchestration · Human-Robot Interaction**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaiarCXicCaibA9q3c9OaC4tYmrhicHmNrlsxicPw8eHibOLCiaDJuJbRSicwTItCwhiabbibkmA2clzSbZ0bjRZIn4CpyToldkEicjQibNTuo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

✨ 用任务路由器协调物理交互与社交对话智能体，将自然语言转化为**有状态机器人技能**。

📖 社交机器人执行自然语言任务，不只要调用 API，还要组合技能、绑定传感器事件并维护对话状态。MistyPilot 用 Task Router 把指令分给 Physically Interactive Agent 与 Social Interaction Agent：前者处理传感器触发和技能调用，后者管理对话状态、多模态响应及结果复用。五组组件测试、Misty 真机执行和 12 人初步研究显示，系统在路由、传感器—技能绑定、状态解析与复用上取得较高准确率，可扩展至 **100 项技能**，方差低于同配置单智能体基线。论文称代码将通过项目页公开。

💡 社交机器人控制需要的不是单个万能代理，而是**按交互类型分工的技能编排**。

🔗 项目链接： 项目页（论文称代码将公开）

🔗 资料来源： https://arxiv.org/pdf/2608.15549

**综合观察**

综合来看，这批论文并没有把“通用机器人”简化成一个更大的策略模型。跨本体物理先验需要统一动作表示，实时执行需要把迭代解码改造成流式缓存，长时程任务需要显式场景状态与规划，而可靠部署还必须面对后门攻击、空间参照歧义和接触反馈。开放资源方面，CLAP 明确开放代码与模型，MILO、FlashVLA、AlloEgo-VLM 等提供代码入口；其余项目以项目页、评测资源或后续开放计划为主，复现时应继续区分“已有仓库”和“代码将发布”。
