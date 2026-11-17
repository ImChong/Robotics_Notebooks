# VTAM：官方源码、复现接口与开放边界

- **官方仓库：** https://github.com/PLAN-Lab/VTAM
- **论文：** https://arxiv.org/abs/2603.23481v2
- **主页：** https://plan-lab.github.io/projects/vtam/
- **核对日期：** 2026-11-17
- **核对范围：** README Installation/Training/Real-robot evaluation/Release status/License；`main.py`、`runner/ge_trainer.py` 的训练入口与步骤；`experiments/eval_xarm.py` 的真机入口及相机、SDK 配置；`configs/real_world/chip_pick_and_place/stage2_vtam.yaml` 的训练数据和模型配置。

## 可运行的公开入口（未实际执行）

1. 安装 `requirements.txt`，按 README 下载 `agibot-world/Genie-Envisioner-v1.0` 的 `GE_base_fast_v0.1.safetensors` 与 `Lightricks/LTX-Video` 的 VAE/tokenizer/text encoder。
2. 训练读取 LeRobot 格式数据：`torchrun --nproc_per_node=4 main.py --config_file configs/real_world/chip_pick_and_place/stage1_world_model.yaml`。在 `main.py` 中动态导入默认 `runner/ge_trainer.py:Trainer`，按 `prepare_dataset → prepare_models → prepare_trainable_parameters → prepare_optimizer → prepare_for_training → prepare_trackers → train` 顺序执行。
3. 将 Stage I checkpoint 写入 Stage II 配置中的 `model_path`，再运行 `torchrun --nproc_per_node=4 main.py --config_file configs/real_world/chip_pick_and_place/stage2_vtam.yaml`；Stage II 的 flow-matching 专家预测动作、本体状态和光流派生的 flow-force proxy。
4. 真机入口：`python experiments/eval_xarm.py --config_file configs/real_world/chip_pick_and_place/deploy_vtam.yaml --ckpt_path <stage2 checkpoint> --ip <xArm IP> --use_third --use_gelsight --prompt "pick up the chip and place it on the plate"`。`eval_xarm.py` 中 `InferenceXArm` 加载模型，`XArmAPI` 建立机械臂连接，`cv2.VideoCapture` 获取 RGB/GelSight，执行模型动作块并以 CSV 记录。需安装 xArm SDK、配置设备相机序号和真机安全措施。
5. `vtam_server.py` 提供通过 TCP 的远程模型服务，用于 UniVTAC 仿真评估；不能把它混为 `eval_xarm.py` 的真机主循环。

## 实际开放与风险

- README 清单勾选训练/推理代码、实机配置与统计、xArm6+GelSight Mini 数据集（https://huggingface.co/datasets/JensenYuan/VTAM_xArm_GelSight）。
- README 明确未开放 UniVTAC 仿真配置和模型 checkpoint；缺 Stage I/II 权重需要先训练，不能宣称直接复现论文。
- **混合许可：** VTAM 原创与部分上游文件 Apache-2.0；Genie Envisioner 衍生部分 CC BY-NC-SA 4.0，不得商用；下载的 LTX/GE 权重还受独立权重许可约束。
- 本核查仅为源码静态检查：没有实际安装训练依赖、执行 GPU 训练、在 UniVTAC 重跑指标或连接真机。
