# Channel Parameters
Time_channel = True
CARRIER_FREQ = 28e9  # 28 GHz mmWave
DELAY_SPREAD = 100e-9  # Fixed moderate delay spread for OTFS variant
L_MIN = 0

# General Parameters
NUM_BITS_PER_SYMBOL = 4
SYMBOL_RATE = 10e6  # 10 MHz bandwidth → 100ns delay resolution
TOT_SYMBOLS_TO_DELIVER = 960  # 15 data × 64 subcarriers → NUM_OFDM_SYMBOL=16

# Single Carrier Parameters
NUM_SC_SYMBOL = TOT_SYMBOLS_TO_DELIVER
T_sc_symbol = 1/SYMBOL_RATE
T_sc_block = T_sc_symbol*NUM_SC_SYMBOL

# OFDM Parameters
FFT_SIZE = 64
CYCLIC_PRFX_LEN = 32
SUBCARRIER_SPACING = int((SYMBOL_RATE/FFT_SIZE))
OFDM_SYMBOLS_FOR_PILOT_INDICES = [0]  # one pilot at symbol 0
NUM_OFDM_SYMBOL = int(TOT_SYMBOLS_TO_DELIVER/FFT_SIZE + len(OFDM_SYMBOLS_FOR_PILOT_INDICES))
T_ofdm_symbol = T_sc_symbol*(FFT_SIZE+CYCLIC_PRFX_LEN)
T_ofdm_block = T_ofdm_symbol*NUM_OFDM_SYMBOL

# Training Parameters
BATCH_SIZE = 10
EBNO_DB_for_training = 10
NUM_TRAINING_ITERATIONS = 500000000000000000000000
SEED = 42

# PER curve Parameters
EBN0_DB_MIN = 0
EBN0_DB_MAX = 25

# OTFS-specific: speed range [m/s]
SPEED_MIN = 0.5    # near-static pedestrian
SPEED_MAX = 120.0   # ~432 km/h high-speed train

# Delay spread range [seconds]
DELAY_SPREAD_MIN = 10e-9
DELAY_SPREAD_MAX = 600e-9  # at 10MHz: 100ns res → DS=600ns spans 6 delay bins

# Time snapshot sampling for OTFS time-varying channel
NUM_TIME_SNAPSHOTS = 24  # 16 symbols × 96 samples/sym = 1536 time samples

# Whether to include PAPR in the loss (multi-task learning)
USE_PAPR_LOSS = False  # pure BCE for stable initial convergence
PAPR_WEIGHT = 5.0
PAPR_THRESHOLD_DB = 0.0
