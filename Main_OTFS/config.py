# Channel Parameters
Time_channel = True
CARRIER_FREQ = 3.5e9
DELAY_SPREAD = 100e-9  # Fixed moderate delay spread for OTFS variant
L_MIN = 0

# General Parameters
NUM_BITS_PER_SYMBOL = 4
SYMBOL_RATE = 1e6
TOT_SYMBOLS_TO_DELIVER = 64

# Single Carrier Parameters
NUM_SC_SYMBOL = TOT_SYMBOLS_TO_DELIVER
T_sc_symbol = 1/SYMBOL_RATE
T_sc_block = T_sc_symbol*NUM_SC_SYMBOL

# OFDM Parameters
FFT_SIZE = 32
CYCLIC_PRFX_LEN = 16
SUBCARRIER_SPACING = int((SYMBOL_RATE/FFT_SIZE))
OFDM_SYMBOLS_FOR_PILOT_INDICES = [0]
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
SPEED_MIN = 3.0     # ~10 km/h pedestrian
SPEED_MAX = 120.0   # ~432 km/h high-speed train
