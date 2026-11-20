# StrokeAIRobotics / Diffusion-Based-Impedance-Learning_T-RO

- **类型：** repo（论文唯一实体的配套代码，不单独创建 wiki 项目节点）
- **代码：** https://github.com/StrokeAIRobotics/Diffusion-Based-Impedance-Learning_T-RO
- **论文：** https://arxiv.org/html/2509.19696v4
- **项目：** https://strokeairobotics.github.io/Diffusion-Based-Impedance-Learning_T-RO/
- **许可证：** MIT（仓库 LICENSE）
- **开放状态：** 已开源，含 Python 模型训练/推理、共享内存 Python 部署接口、KUKA LBR iiwa 的 C++ 控制客户端、AVP 遥操作和示例 Parkour 数据。
- **核对范围：** 已检查 README、ImpedanceLearning/DiffusionModel.py、data.py、models.py、train_val_test.py、stiffness_communication.py 以及 Deployment/RobotController/src/MyLBRClient.cpp 入口及 README 操作说明；属于代码静态审阅，未实际安装、运行或上机器人。
- **通读：** 已完成
- **wiki：** [论文唯一详情](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md)

## 运行入口与真实职责

| 路径 | 功能 |
|---|---|
| ImpedanceLearning/DiffusionModel.py | 载入并按轴归一化示例轨迹，80/10/10 拆分，构建 DataLoader 与模型，依次训练、测试、应用推理仿真 |
| ImpedanceLearning/data.py | ImpedanceDatasetDiffusion 与 load_robot_data，提供位置/姿态、力矩、速度与动力学量 |
| ImpedanceLearning/models.py | NoisePredictorTransformerWithCrossAttentionTime；用外力/外力矩条件化 Transformer |
| ImpedanceLearning/train_val_test.py | train_model_diffusion、test_model、inference_simulation、deployment；含旋转噪声与刚度估计工具调用 |
| ImpedanceLearning/stiffness_communication.py | 加载 stats.pt + best_model.pth，从共享内存读取 16 帧末端状态及 wrench，调用 deployment 估计六轴刚度并将结果写回共享内存 |
| AVPTelemanipulation/avp_stream/VisionProCPPCommunication.py | Vision Pro 遥操作数据与 C++ 控制端的通信入口 |
| Deployment/RobotController/src/MyLBRClient.cpp | C++ FRI 任务空间力矩控制端；采集反馈、与 Python 共享内存交互、应用增益（需硬件配置） |
| Deployment/TrajectoryGeneration/ | 名义 ZFT 关键点插值与轨迹生成 |

## README 最短离线复现

```bash
git clone https://github.com/StrokeAIRobotics/Diffusion-Based-Impedance-Learning_T-RO.git
cd Diffusion-Based-Impedance-Learning_T-RO
conda env create -f environment.yml
conda activate ImpedanceLearning
python ImpedanceLearning/DiffusionModel.py
```

> README 内示范的 cd DiffusionBasedImpedanceAdaptation 与当前 GitHub 克隆目录名不同；以上路径按仓库实际名称改写。训练文件有本机数据路径和可调参数，运行前还须手动核查数据目录与依赖。静态审阅不等于已验证脚本无错。

## 真机复现注意

- KUKA LBR iiwa 控制走 FRI 5 ms（200 Hz），C++ 侧使用硬件传感器和共享内存；Python 环境需可读模型文件，示例代码硬编码了 stats / checkpoint 的绝对路径。
- 按 README 逐项填补 MyLBRClient.cpp 中 TODO[FT]、TODO[CONFIG]、TODO[GAINS]、TODO[IPC]、TODO[PY] 等；核对机器人姿态、坐标系、工具几何、F/T 极性、共享内存结构及安全限值。
- 与论文实验一致性核查：力/力矩测量频率、滑窗 16 帧、强制限速、刚度上下界、力矩限幅、模型推理周期及外部停止条件。
- 源码只是参考实现；不能把公开代码视为可直接在其他机器人上安全上线的通用控制器。

## 对 wiki 的映射

- [论文唯一节点及运行时序图](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md)
- [原文核对](../papers/diffusion-based-impedance-learning-2509-19696.md)
- [作者项目主页](../sites/diffusion-based-impedance-learning.md)
