# PAFF_code 官方实现核查

- **代码：** https://github.com/geyuying/PAFF_code
- **固定提交：** dc84b2021dc20f7a7463c52ccbb4d9649376a8b8
- **论文：** [v4 归档](../papers/paff_2212_07398.md)；[项目页](../sites/paff.md)。
- **核对范围：** README 五阶段流程；Play/hulc/evaluation/evaluate_policy_record.py 的 main、rollout；CLIP_Finetune、Relabel、Policy_Finetune 的 hulc/training.py；Relabel/conf/config.yaml、conf/model/clip_ft.yaml 与 hulc/models/hulc_clip_record_neg.py 的 training_step。静态核查相关入口，未全仓审计或运行训练。
- **通读：** 未完成
- **开放范围：** README 明确提供 CALVIN A/B/C → D 适应训练代码；数据按 HULC 指引获取。未据此认定桌面/真机实验全部可直接复现。

## 实际运行接口

1. Play 的 evaluate_policy_record.py 调用策略 step 后将动作、静态/手爪相机、机器人状态写 npz，再 env.step；save_dir 与文件名直接拼接，需检查输出路径。
2. CLIP_Finetune/hulc/training.py 通过 Hydra 实例化 datamodule、model 与 Lightning Trainer，调用 fit。
3. Relabel 默认 clip_ft 配置实例化 hulc_clip_record_neg.Hulc；training_step 计算视觉与候选语言相似度，argmax 后以 score > 0.0 且 ann_index != 34 筛选，追加写 hulc/record_D_relabel.txt。旁支 hulc_clip_record.py 有 3.0 阈值，不可混同默认配置。
4. Policy_Finetune/hulc/training.py 初始化时含历史 runs/.../epoch=8.ckpt 路径；之后 Trainer.fit。README 要求输入 record_D_after_relabel，阶段间数据整理须补齐。

## 映射

[唯一论文节点与时序图](../../wiki/entities/paper-paff-2212-07398.md)。源码尚未全读完及实验未运行，已明确记录；论文正文通读状态与代码全仓审计分别记载。
