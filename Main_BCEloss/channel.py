from sionna.phy.channel.tr38901 import AntennaArray, CDL, TDL
from sionna.phy.channel import CIRDataset
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
from utils.DebugDeltaChannel import DebugDeltaChannel
from utils.TDL_RandomDS import TDL_RandomDS
from config import *

# CDL channel
delay_spread = DELAY_SPREAD
direction = "uplink"  
cdl_model = "B"       
speed = 0            
ut_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQ)
bs_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQ)
cdl = CDL(cdl_model, delay_spread, CARRIER_FREQ, ut_array, bs_array, direction, min_speed=speed)

# TDL channel
delay_spread = DELAY_SPREAD
direction = "uplink"  
tdl_model = "A"       
tdl = TDL(model = tdl_model, delay_spread = delay_spread, carrier_frequency = CARRIER_FREQ, min_speed = 0.0, max_speed = 0.0)

# TDL random delay spread channel
delay_spread_min = 10e-9        #[Sec]
delay_spread_max = 600e-9    #[Sec]
direction = "uplink"  
tdl_model = "A"       
tdl_randomDS = TDL_RandomDS(model = tdl_model, delay_spread_min = delay_spread_min, delay_spread_max = delay_spread_max, carrier_frequency = CARRIER_FREQ, min_speed = 0.0, max_speed = 0.0)

# RBF channel
rbf = RayleighBlockFading(num_rx=1, 
                        num_rx_ant=1, 
                        num_tx=1, 
                        num_tx_ant=1)

# DebugDeltaChannel
debug_channel = DebugDeltaChannel(1, 1, 1, 1)

# Chosen channel to use:
# channel_model = tdl
# channel_model = debug_channel
channel_model = tdl_randomDS