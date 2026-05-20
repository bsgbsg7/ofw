from sionna.phy.channel.tr38901 import TDL
from config import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from TDL_RandomDS import TDL_RandomDS

# Time-varying TDL channel with random speed (Doppler) and fixed moderate delay spread
# Speed varies per batch sample → network sees both low and high Doppler
# Low Doppler → channel is nearly static → OFDM-like waveform sufficient
# High Doppler → channel varies rapidly across symbols → OTFS-like waveform needed

delay_spread_min = 50e-9
delay_spread_max = 300e-9
tdl_model = "A"

channel_model = TDL_RandomDS(
    model=tdl_model,
    delay_spread_min=delay_spread_min,
    delay_spread_max=delay_spread_max,
    carrier_frequency=CARRIER_FREQ,
    min_speed=SPEED_MIN,
    max_speed=SPEED_MAX
)
