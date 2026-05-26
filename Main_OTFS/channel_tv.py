"""
时变 TDL 信道模块 (Time-Varying TDL Channel)
===============================================
本模块基于 3GPP TR38.901 规范的抽头延迟线 (Tapped Delay Line, TDL) 信道模型，
构建一个具备随机多普勒频移（随机速度）和固定中等时延扩展的时变信道。

核心设计思想：
  - 每个 batch 样本的速度随机采样 → 网络同时见到低多普勒和高多普勒场景
  - 低多普勒 → 信道近似静态 → OFDM 类波形即可胜任
  - 高多普勒 → 信道在符号间快速变化 → 需要 OTFS 类波形处理

使用的信道模型封装：
  TDL_RandomDS 是对 Sionna 库中 TDL 模型的扩展，支持：
    1. 时延扩展 (delay spread) 在 [min, max] 范围内随机采样（每 batch 不同）
    2. 速度 (speed / Doppler) 在 [min, max] 范围内随机采样（每 batch 不同）
"""

# 从 Sionna 物理层信道模块导入标准 TDL 模型（此处仅作参考，实际使用的是下面的 TDL_RandomDS）
from sionna.phy.channel.tr38901 import TDL

# 导入全局配置文件中的常量：载波频率、速度范围等
from config import *

import sys
import os

# 将 utils 子目录添加到 Python 路径，以便导入自定义的 TDL_RandomDS 模块
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from TDL_RandomDS import TDL_RandomDS

# ============================================================================
# 信道参数配置
# ============================================================================

# 时延扩展 (RMS delay spread) 范围 [秒]
# 最小值 50ns —— 对应短时延场景（如室内）
# 最大值 300ns —— 对应中等时延场景（如城市微小区）
# 该范围适中，既不过于简单（极短时延），也不过于恶劣（长时延），
# 适合训练网络在中等多径条件下的鲁棒性
delay_spread_min = 50e-9   # 50 纳秒
delay_spread_max = 300e-9  # 300 纳秒

# TDL 模型类型："A" 表示 TDL-A 模型
# TDL-A 是 3GPP TR38.901 中定义的非视距 (NLoS) 模型，具有 23 条多径
tdl_model = "A"

# ============================================================================
# 实例化时变信道模型
# ============================================================================

# TDL_RandomDS 参数说明：
#   model              - TDL 模型类型（"A"/"B"/"C"/"D"/"E" 等）
#   delay_spread_min   - 每次 batch 随机采样的时延扩展下限 [秒]
#   delay_spread_max   - 每次 batch 随机采样的时延扩展上限 [秒]
#   carrier_frequency  - 载波频率 [Hz]，从 config 导入
#   min_speed          - 最小移动速度 [m/s]，决定最小多普勒频移
#   max_speed          - 最大移动速度 [m/s]，决定最大多普勒频移
#
# 注意：速度在每个 batch 样本中随机均匀采样，使得网络在训练过程中
#       暴露于不同多普勒条件下的信道，从而学习到自适应的波形策略
channel_model = TDL_RandomDS(
    model=tdl_model,
    delay_spread_min=delay_spread_min,
    delay_spread_max=delay_spread_max,
    carrier_frequency=CARRIER_FREQ,
    min_speed=SPEED_MIN,
    max_speed=SPEED_MAX
)
