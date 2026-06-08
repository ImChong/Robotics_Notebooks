# Go2 Motion Imitation

> 来源归档

- **标题：** Go2 Motion Imitation
- **类型：** repo
- **链接：** https://github.com/TSUITUENYUE/motion-imitation
- **仿真器：** Genesis
- **入库日期：** 2026-06-08
- **一句话说明：** Unitree Go2 运动模仿管线：MoCap/视频骨架 → `retarget_motion` 转 Genesis 格式 → 关节速度匹配 RL 训练。
- **沉淀到 wiki：** 是 → [`wiki/entities/go2-motion-imitation.md`](../../wiki/entities/go2-motion-imitation.md)

## 重定向入口

```bash
python retarget_motion/retarget_motion.py --input_file=source_motion.txt --output_file=retargeted_motion.txt
```
