---
title: 去年被BeyondMimic统治，今年被SuperMimic：还是同一个第一作者，论文出来几天，开源项目已经遍地走了……
author: 具身智能研究室（Yuanxq）
date: "2026-10-30 01:14:00"
source: "https://mp.weixin.qq.com/s/-6fckwPa2Bc8-FhMAzP8Tg"
fetch: scripts/fetch_wechat_article.py（Chromium + MicroMessenger UA，纯文本）
---

去年被BeyondMimic统治，今年被SuperMimic：还是同一个第一作者，论文出来几天，开源项目已经遍地走了……
Original Yuanxq 具身智能研究室
 Oct 30, 2026, 1:14 AM

学术圈还是太卷了……supemimic论文刚出来，已经有人用一小时写出复现，紧接着各种改进、移植和部署项目就接连冒出来了。

论文题目：SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model
机构：加州大学伯克利分校
项目链接：https://supermimic.github.io/

截至 2026 年 10 月 29 日，以下整理已检索到的 42 个复现及相关扩展项目，每项附仓库链接和一句说明。

基础复现

supermimic_vibed（Star：716）｜https://github.com/zita-ch/supermimic_vibed｜基于 Isaac Lab 3 与 MuJoCo Warp 的 G1 复现，提供 LAFAN1 训练、完整动作评估和渲染工具。

SuperMimic-Reproduced（Star：315）｜https://github.com/CuboiLeo/SuperMimic-Reproduced｜基于 mjlab 从论文重新实现，包含 LAFAN1、BONES-SEED 数据流程、多卡训练、中文实验记录和预训练权重。

SuperMimic Mjlab（Star：8）｜https://github.com/Breezewrf/supermimic_mjlab｜面向 G1 的 mjlab 复现，公开归一化、参考朝向等消融实验，以及策略和世界模型权重。

supermimic_vibe（Star：5）｜https://github.com/dhmbb2/supermimic_vibe｜基于 mjlab 复现 G1 跟踪，包含 PPO-Sim、PPO-WM、想象时域、价值网络和抗扰实验。

Supermimic_mod（Star：11）｜https://github.com/YanYuan98/Supermimic_mod｜以 CuboiLeo 版本为基础，调整参考朝向、输入归一化和训练执行效率，附中文实现说明与实验记录。

realmimic（Star：0）｜https://github.com/kevinhenryzz/realmimic｜面向单张 RTX 4090 的大动作库复现，公开模型权重、评估工具，并接入 SONIC 部署运行时。

RaraAlu/Supermimic（Star：5）｜https://github.com/RaraAlu/Supermimic｜基于 mjlab 的复现，支持 LAFAN1 与原生 50Hz HDF5 动作数据，包含多卡训练和性能优化工具。

方法改进

InstantTrack（Star：52）｜https://github.com/Huy-Quang-Dao/instantTrack｜结合 GPU 训练循环优化、策略滑动平均和小型世界模型，报告单张 RTX 5090 约 15 分钟完成 LAFAN1 训练目标。

allmimic（Star：31）｜https://github.com/Shimingyi/allmimic｜基于 supermimic_vibe，改进上下半身动作表示、世界模型损失权重与动作采样。

Rethinking SuperMimic（Star：27）｜https://github.com/btx0424/rethinking-supermimic｜研究目标条件怎样进入 G1 策略网络，对比直接拼接与 FiLM 特征调制，提供实现对照和回放工具。

SuperMimic on LocoMuJoCo（Star：5）｜https://github.com/Alchedata/supermimic-locomujoco｜在 LocoMuJoCo 中训练 H1 跟踪器，探索 ODE 世界模型和动作输出防饱和正则。

Standing MBRL vs PPO（Star：20）｜https://github.com/JackMin7/supermimic_whole_body_tracking｜在同一 G1 静态站立任务上对比 SuperMimic 风格世界模型学习与 BeyondMimic 的 PPO 训练。

本体与硬件

MorphMimic（Star：152）｜https://github.com/lzyang2000/morphmimic｜让宇树 G1、智元 X2 Ultra 和加速进化 T2 共享世界模型与策略，支持完整动作、五点目标和遥操作接口。

SuperMimic Retargeted（Star：9）｜https://github.com/tianshuwu/supermimic-retargeted｜将 CuboiLeo 版本扩展到 G1、A3 和 X2 Ultra，补充 SOMA 重定向、坐标修正和足部接触检查。

SuperMimic E1（Star：5）｜https://github.com/wenconggan/SuperMimic_e1｜将 Isaac Lab 3 实现适配到 21 自由度 E1，提供原生 MuJoCo 回放与仿真迁移评估。

SuperMimic Oli（Star：3）｜https://github.com/liuskywalkerjskd/supermimic_oli｜将 supermimic_vibed 移植到逐际动力 31 自由度 Oli，包含动作重定向、训练和视频导出。

SuperMimic on RTX 5080（Star：2）｜https://github.com/rzc156/supermimic-rtx5080｜面向单张 16GB RTX 5080 的 mjlab 复现，提供 LAFAN1 训练配置、逐动作评估与演示。

SuperMimic on Apple M5（Star：1）｜https://github.com/WXyuany/supermimic-m5｜使用 CPU MuJoCo 与 PyTorch MPS 在 Apple M5 上训练，公开运行流程、模型权重和逐动作结果。

SuperMimic-MuJoCo（Star：15）｜https://github.com/panhainan2020/SuperMimic-MuJoCo｜将 supermimic_vibed 移植到独立 MuJoCo 与 mjlab，提供训练、推理权重及 LAFAN1、100STYLE 评估。

SuperMimic Wheel-Legged（Star：7）｜https://github.com/RadiantForever/supermimic_wheelbipe｜面向 wheelbipe 轮腿机器人，将参考动作库改为速度指令库，训练腿部与轮子的协同控制。

感知与地形

InstinctMJ · SuperMimic（Star：301）｜https://github.com/project-instinct/InstinctMJ/tree/supermimic｜将 SuperMimic 接入 InstinctMJ，研究深度感知驱动的 G1 动作跟踪与跑酷。

Go2-SuperMimic（Star：69）｜https://github.com/Zhengtianjin/go2-supermimic｜适配宇树 Go2 的追逐、躲避与地形任务，公开平地、障碍和地形策略及评估工具。

SuperMimic Parkour（Star：12）｜https://github.com/yhx1203/supermimic_parkour｜结合 PHP 跑酷动作与地形数据，在 mjlab 中训练 G1 五类跑酷技能，提供权重与浏览器回放。

SuperMimic-perceptive（Star：6）｜https://github.com/mode1223345/SuperMimic-perceptive｜基于 Isaac Lab 3，让策略、价值网络和世界模型共享预训练的高度图编码，探索地形感知控制。

操作与灵巧手

SuperMimic for Robot Manipulation（Star：39）｜https://github.com/emoPointer/supermimic_manipulation｜将方法迁移到 ARX5 机械臂，提供 Lemon、BoxInBin、MugHang 三类操作任务、权重与重置状态数据。

Dex SuperMimic（Star：74）｜https://github.com/CumulusAlpha/Dex-Supermimic｜面向无极灵巧手的方块重定向与转笔，提供三类任务、预训练权重和可视化工具。

In-Hand Rotation mjlab（Star：34）｜https://github.com/Msornerrrr/in-hand-rotation-mjlab｜在 LEAP Hand 上对比世界模型学习与 PPO，覆盖方块旋转和 22 类物体的指尖旋转，发布训练权重。

Humanoid Mimic · SuperMimic（Star：7）｜https://github.com/shuayyy/humanoid-amp-ppo｜在 mjlab 的 G1 行走、下蹲和搬箱任务中，对比 SuperMimic 与 PPO-Sim。

肌肉与交互

muscle-supermimic（Star：33）｜https://github.com/kaibohex/muscle-supermimic｜将方法用于 MS-Human-700 和 MyoFullBody 肌肉骨骼模型，提供可微肌肉计算、动作重定向与全身跟踪。

SuperMimic-PERA（Star：0）｜https://github.com/shangchaohu718/supermimic-pera｜基于 UniSuperMimic，围绕 G1 动作跟踪继续探索身体交互与滑板等任务。

SuperMimic Backflip Diagnostics（Star：2）｜https://github.com/liudi159/supermimic-backflip-5090｜聚焦单张 RTX 5090 上的 G1 后空翻，记录接触力、身体穿透、腿部对称性和落地稳定性。

框架集成

UniSuperMimic（Star：204）｜https://github.com/Motphys/Uni-SuperMimic｜在 UniLab 中复现 SuperMimic，提供 LAFAN1、BONES-SEED 数据管线，以及多仿真后端入口和评估工具。

ScaleBFM_Plus · SuperMimic（Star：19）｜https://github.com/Hxxxz0/ScaleBFM_Plus｜将世界模型与想象轨迹策略训练加入 ScaleTrack，结合原有 Transformer 控制模型，提供训练、权重和导出流程。

HoloMotion · SuperMimic（Star：749）｜https://github.com/HorizonRobotics/HoloMotion｜将 SuperMimic 接入 HoloMotion 的 G1 资产、HDF5 动作数据和训练流程，支持 MuJoCo Warp 与 Isaac PhysX。

TrackerLab SuperMimic（Star：13）｜https://github.com/Renforce-Dynamics/trackerlab-supermimic｜基于 Isaac Lab 与 PhysX，提供多机器人重定向、训练、完整动作评估和策略导出流程。

SuperMimic Lab（Star：2）｜https://github.com/chu-fa/isaaclab-supermimic｜将算法封装为可复用的 Isaac Lab 学习器与训练器，提供自定义任务接口及智元 A3 示例。

SuperMimicKit（Star：4）｜https://github.com/OMEGA-i/SuperMimicKit｜面向 G1、H2 和 A3 的多本体工具包，集成重定向、训练、策略与世界模型导出及 WebGPU 预览。

真机与部署

SuperMimic-Repro-Sim2Real（Star：36）｜https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real｜南京大学团队基于 CuboiLeo 版本，补充视频转动作、舞蹈训练与 G1 部署，公开真机演示和预训练模型。

SuperMimic G1 Deployment（Star：8）｜https://github.com/KyrieYoung926/supermimic_deploy｜为 SuperMimic-Reproduced 策略提供 ONNX Runtime、MuJoCo 与 Unitree SDK2 运行流程，包含仿真和 G1 真机演示。

相关探索

FlatWorld · SuperMimic-2D（Star：43）｜https://github.com/swaggyliu/FlatWorld｜在二维人形角色中实现共享世界模型与短时反向传播训练，展示行走、跑步、跳跃和倒地起身等技能。

Ref2Dex · Dex Contact MBRL（Star：暂不可查）｜https://github.com/hitsz-oyx/Ref2Dex-agent｜社区目录收录的 Inspire 灵巧手接触动力学与短时策略优化实验；本次检索时仓库链接返回 404。

G0.5 · SuperMimic-style Steering（Star：0）｜https://github.com/azizkhadraoui/g05｜在 G0.5 的 VLA 实验中借鉴 SuperMimic 思路，利用世界模型和价值梯度训练动作生成的引导策略。

Like Author

 给作者赏杯咖啡 

宇树相关 · Table of Contents
Previous
ETH RSL人形loco-manip新进展：不用动作示范，宇树G1用一个策略学完整套搬运
​
具身智能研究室
Like
Share
Boost
Comment