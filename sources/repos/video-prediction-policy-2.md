# Video Prediction Policy 2（VPP2）官方代码仓库

> 来源归档（ingest · 官方 GitHub 仓库）

- **标题：** Video Prediction Policy 2: Predict Better, Act Better
- **类型：** repo（官方实现）
- **代码：** <https://github.com/roboterax/video-prediction-policy-2>
- **项目页：** <https://robert-gyj.github.io/video-prediction-policy-2/>
- **论文：** <https://arxiv.org/abs/2610.10270>（v2：2026-10-08）
- **许可证：** MIT（代码）；模型权重、数据和模拟器资产另有各自条款
- **核查日期：** 2026-11-19
- **仓库快照：** 官方公开 Python 项目，默认分支 main；star / fork 等动态数字不作为恒定项目事实
- **一句话说明：** VPP2 官方训练、推理与基准评测实现，面向 RoboDojo 与 LIBERO 系列操作任务；与论文和项目页合并归入同一个 wiki 实体节点。
- **沉淀到 wiki：** [Video Prediction Policy 2（VPP2）](../../wiki/entities/video-prediction-policy-2.md)

## 仓库入口

- `src/vpp2/models/wan21_14b/`：Wan2.1-14B 视频骨干、VPP2、MoT 与 Action DiT 实现。
- `src/vpp2/`：数据、训练、权重、策略推理及基准运行支持。
- `policy/VPP2/`：部署适配器；README 提供 VPP2 policy server 入口。
- `scripts/robodojo/`：数据准备、文本缓存、初始化、联合训练、checkpoint 导出、服务和闭环评估。
- `scripts/libero/`：Video / Action 阶段训练、标准 LIBERO、OOD 与 PRO 评测。
- `configs/robodojo/`、`configs/libero/`：对应实验设置。

## 安装与复现线索

README 推荐 Python 3.10。RoboDojo 文档给出的已测试策略环境使用 PyTorch 2.11.0、torchvision 0.26.0 和 CUDA 13.0；训练依赖还包括 DeepSpeed。RoboDojo / Isaac Sim 放在独立模拟器环境中。各脚本直接从 checkout 加载 `src/vpp2`，无需先安装项目包。

```bash
git clone https://github.com/roboterax/video-prediction-policy-2.git
cd video-prediction-policy-2
conda create -n vpp2 python=3.10 -y
conda activate vpp2
python -m pip install -r requirements-train.txt -c environment-reference.txt
```

再按所选基准阅读对应指南：

- [RoboDojo 训练与闭环评测](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/robodojo.md)
- [LIBERO / OOD / PRO 训练与评测](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/libero.md)
- [数据及训练约定](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/training.md)
- [评测协议](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/evaluation.md)

## 模型与资产开放范围

README 链接公开的 [Hugging Face 模型仓库](https://huggingface.co/Haodong082399/VPP2) 与 [ModelScope 模型仓库](https://modelscope.cn/models/haodong123/VPP2_preview)。官方还发布了 Stage-1（49 帧）、Stage-2（17 帧）视频 checkpoint，RoboDojo history-conditioned Video-10k 及可评测的成对 Video + Action2B checkpoint；LIBERO 发布 Video-10k + Action-30k。ModelScope 下载需要有权限的账号。

**不是所有训练资产都已开放：**

- README 在 2026-10-10 发布 Stage-1/2 视频模型权重和预测指南；10-12 发布 RoboDojo history-conditioned Video-10k 训练流程。但其注释的 TODO 仍包含大规模 event-level 视频预训练代码和配置，尚不能据此完成论文所有阶段的从零复现。
- RoboDojo 3500 条示范可从 RoboDojo 官方公开 EE16 LeRobot v3 数据集按指定 revision 下载，再通过 [training.md](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/training.md) 给出的转换流程生成训练数据；这不等于大规模 event-level 预训练原始数据全部开放。
- 第三方组件保留原许可证；权重、数据集和仿真资产需分别核对适用条款。

## 官方 README 关键事实

- 仓库描述为论文 **Video Prediction Policy 2: Predict Better, Act Better** 的官方实现。
- 主线方法为 event-level manipulation video 预训练、固定 horizon 单步视觉规划器，以及通过 mixture-of-transformers 学习隐式逆动力学的动作专家。
- RoboDojo 和 LIBERO 系列分别有配置、训练/推理入口与评估说明；README 报告的 benchmark 结果须结合论文协议读取。
- README 的 2026-10-08 News 称其在官方 RoboDojo-Sim leaderboard 位列第一，并列出 39.26 score / 32.26% success rate；该排名是日期快照，不代表未来榜单状态。
- README 的 2026-10-10 / 10-12 News 分别发布视频预测 checkpoint 和 RoboDojo history-conditioned 视频训练代码，具体命令以仓库现行文档为准。

## 关联归档

- [官方项目页归档](../sites/video-prediction-policy-2.md)
- [论文归档：arXiv:2610.10270](../papers/vpp2_arxiv_2610_10270.md)
- [合并后的 wiki 实体节点](../../wiki/entities/video-prediction-policy-2.md)
