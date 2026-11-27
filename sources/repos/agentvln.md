# AgentVLN 官方代码与训练数据归档

- **项目页：** https://allenxinn.github.io/AgentVLN/
- **论文：** [预印本](https://arxiv.org/abs/2603.17670v1) · [ECCV 2026](https://doi.org/10.1007/978-3-032-37167-6_7)
- **代码：** [Allenxinn/AgentVLN](https://github.com/Allenxinn/AgentVLN) · [核对提交](https://github.com/Allenxinn/AgentVLN/tree/c1c63bd97d1c54c5694af3d61c63285001525439)
- **数据集：** [AgentVLN-Instruct 数据卡](https://huggingface.co/datasets/allenxinn/AgentVLN-Instruct)，按场景 LMDB 存图像，以 `exploration_data.json` 保存逐步路点、可见性、位姿坐标和动作。
- **公开边界（2026-11-27）：** 有数据生成、训练、Habitat 评估、第三方技能和安装支架资料；README 的预训练 checkpoint 与环境安装指南仍未勾选。没有据此宣称整套真机工程可一键复现。
- **评估入口：** `agentvln/eval/eval_habitat.py` 的 `main()` 调用 `run_habitat_eval`；默认 YAML 仍需填数据、场景和 checkpoint 路径。代码用 `PromptBuilder`、`ModelInference` 和 `ResponseParser` 接起像素目标或原子动作分支，未见有独立的多轮语言深度查询处理分支。
- **本地冒烟：** 上游 `compute_ndtw` 用合成轨迹做 5 项断言（同轨、连续重复点、已知偏移、零阈值、非有限阈值）；未启动 Habitat、加载模型、下载整套数据或连接实机。
- **对应知识页：** [AgentVLN](../../wiki/entities/paper-agentvln.md)
