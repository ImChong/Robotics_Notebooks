# Long-WAM: Scaling the Context of World-Action Models

> 来源归档（paper；核对 arXiv v1、项目页和官方代码文档；2026-10-08）

- **arXiv：** <https://arxiv.org/abs/2610.10528> · [HTML v1](https://arxiv.org/html/2610.10528v1) · [PDF](https://arxiv.org/pdf/2610.10528)
- **提交时间：** 2026-10-07（v1）
- **项目页：** <https://nvlabs.github.io/LongLive/Long-WAM/>
- **代码：** <https://github.com/NVlabs/LongLive/tree/main/Long-WAM>（LongLive monorepo 子目录）
- **模型：** <https://huggingface.co/collections/Efficient-Large-Model/long-wam>
- **作者：** Wei Huang, Bohan Zhang, Chenzhi Liu, Isabella Liu, Shuai Yang, Weian Mao, Luozhou Wang, Yicheng Xiao, Weifeng Lin, Qixin Hu, Bryan Chu, Sifei Liu, Linxi Fan, Xiaojuan Qi, Song Han, Yukang Chen
- **机构：** NVIDIA, MIT, The University of Hong Kong, UC San Diego；Wei Huang 与 Bohan Zhang 共同第一作者。
- **研究问题：** 因果 WAM 如何使用更长观测历史，而不同比例扩展未来预测与动作输出长度。
- **方法：** 从 LongLive-2.0-Robot 自回归视频预测模型继续适配；因果视频专家预测未来视觉 latent，动作专家利用长历史与未来 latent 生成动作块。视频专家不读取动作 token，推理无需解码完整 RGB 视频。

## 核心结果摘录

1. RoboCasa GR-1 从 0 秒历史时 SR 63.3% 提升至 19.2 秒历史时 78.7%；项目页另列 2.4 秒时 66.3%，应按配置区分。
2. 论文报告 LIBERO-Long 99.5% SR、RoboTwin 2.0 94.4% SR，均限于各自 benchmark 协议。
3. 动态杯叠放真机任务报告 19/20 成功；有限任务与样本数不能替代广泛验证。
4. RTX 5090 上 107.4 ms/action chunk，项目方称包含未来 latent 预测；不是本知识库独立测量值。
5. 历史窗口配置包含 0、2.4、4.8、9.6、19.2、38.4 秒。长窗口回落与轨迹平均长度和 padding 有关，不足以单独证明固定记忆上限。

## 训练和解释边界

项目页称 LongLive2.0-Robot 从 LongLive-2.0 checkpoint 继续训练，汇集约 10,000 个窗口等价小时的机器人与第一视角视频（RoVid-X、AgiBot World、EgoDex、EgoVerse、VITRA）。先进行不需动作标签的未来视频预测适配，再接入动作生成。

- 页面上的 future prediction 视频是模型预测，不应默认当作真实机器人 rollout。
- 源码 VERIFICATION 记录 CPU 测试/CLI 检查，不覆盖 GPU benchmark、完整仿真重跑或真机任务；BENCHMARKS 标明完整复现尚未验证。
- 不同模型卡的本体、相机、状态/动作表示与动作块可能不同。
- 以上数字是作者报告值；该归档没有独立重跑实验。

## 关联知识节点

- [Long-WAM 独立项目节点](../../wiki/entities/paper-long-wam-scaling-context.md)
- [WAM 概念页](../../wiki/concepts/world-action-models.md)
