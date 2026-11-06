# UniSim 论文通读（生成式世界模型视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/generative-world-models.md`）

- **标题：** Learning Interactive Real-World Simulators (UniSim)
- **作者：** Sherry Yang、Yilun Du、Seyed Kamyar Seyed Ghasemipour 等（UC Berkeley / Google DeepMind / MIT / Alberta）
- **类型：** paper / world-model / simulator
- **原文：** <https://arxiv.org/abs/2310.06114>（2026-11 通过 arXiv PDF 获取全文，v3，ICLR 2024）
- **一句话说明：** 把多源数据统一为「动作进、视频出」接口，训练视频扩散观测预测模型作为通用交互模拟器，并用它训练 VLM 策略、RL 策略与字幕模型。
- **核对范围：** 通读 §1–§6（数据整合、观测预测与扩散参数化、模拟结果、三类应用、相关工作、局限）；附录（数据集清单、训练细节、更多真机结果）仅浏览，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **模拟器定义（§2）：** 给定世界状态（如一帧图像）与动作，输出该动作的视觉后果（视频）。
- **数据整合（§2.1）：** 将模拟渲染、真机数据、人类活动视频、全景扫描、互联网图文对的动作统一为文本（T5 嵌入）并拼接离散化的低层控制；图像当作单帧视频、图注当作动作。
- **观测预测（§2.2）：** 建模 p(o_t | h_{t-1}, a_{t-1})，用视频扩散参数化，历史帧与噪声按通道拼接，动作用 classifier-free guidance 注入；可自回归 rollout 做长程模拟。架构为视频 U-Net，5.6B 参数，512 块 TPU-v3 训练 20 天。
- **历史条件消融（§3.2）：** Ego4D 上条件取 4 帧近期历史优于 1 帧，过远的历史反而变差，超过 4 帧无进一步提升；低数据域加数据集标识符可提升质量，但损害跨域泛化。
- **长程模拟（§3.1）：** 连续 8 次语言指令的开关抽屉操作中，先前放入的物体被保持。
- **应用一：长程 VLM 策略（§4.1）：** 用模拟 rollout 做 hindsight relabeling 生成 1 万条长程轨迹，Language Table 上 RDG 指标约为原短程数据训练的 3–4 倍（表 2）。
- **应用二：RL（§4.2）：** 以模拟器 rollout 加学习到的奖励模型，用 REINFORCE 微调 VLA；模拟内评估成功率 0.58→0.81（表 3，由视频 rollout 定性判定），并可零样本部署到真机。
- **应用三：字幕（§4.3）：** 仅用 UniSim 生成视频微调 PaLI-X，ActivityNet CIDEr 15.2→46.23，约为真实数据微调的 84%。
- **局限（§6）：** 幻觉（对不合理动作，如让桌面机器人「洗手」，场景会变形）；记忆有限（仅条件于近期若干帧，抽屉里的苹果打开后可能消失）。
- **适用范围（脚注 1）：** 「universal」仅指统一的动作-视频接口，不模拟声音等。

## 对 wiki 的映射

- [`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2310.06114>
