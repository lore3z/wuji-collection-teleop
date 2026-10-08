============================================================
V9.4 GOOD ROOT-ONLY PINCH BASELINE
============================================================

当前确认效果：

1. MuJoCo 四指对指效果正确
2. 实体 L20 四个对指终点正确
3. 对指识别突变已明显减小
4. 拇指会提前自然向预备位置运动
5. PREP 阶段仍保留实时手套控制
6. 最终 CLOSE 阶段：
   - 拇指姿态锁定
   - 目标手指侧摆/PIP锁定
   - 仅目标手指 MCP 根部完成捏取
7. 从完全捏合稍微张开时：
   - 拇指不会明显横摆
   - 主要只打开目标手指 MCP 根部

------------------------------------------------------------
运行参数
------------------------------------------------------------

V94_REAL_THUMB_ROLL_BIAS=0.0
V94_REAL_THUMB_YAW_BIAS=0.0
V94_REAL_THUMB_PITCH_BIAS=0.0
V94_REAL_THUMB_MCP_BIAS=0.0

V94_PINCH_DETECT_ON_MM=90
V94_PINCH_LOCK_DWELL_MS=120
V94_PINCH_DOMINANCE_MM=4

V94_PINCH_PREP_ON_MM=70
V94_PINCH_COMMIT_ON_MM=55
V94_PINCH_ROOT_ONLY_ON_MM=45
V94_PINCH_CLOSE_DONE_MM=25
V94_PINCH_RELEASE_MM=100

V94_PINCH_ANTICIPATE_GAIN=0.15
V94_PINCH_PREP_GAIN=0.65
V94_PINCH_ROOT_PREBEND=0.20
V94_PINCH_ASSIST_SLEW_S=0.18

------------------------------------------------------------
状态机
------------------------------------------------------------

FREE
  ↓
ANTICIPATE
  ↓
PREP
  ↓
COMMIT
  ↓
ROOT_ONLY
  ↓
HOLD

ROOT_ONLY:
thumb = locked
finger roll/PIP = locked
only target MCP/root changes

============================================================
DO NOT MODIFY THIS BACKUP
============================================================
