# Channel Parameters
Time_channel = True
CARRIER_FREQ = 3.5e9
DELAY_SPREAD = 600e-9 #100e-9
L_MIN = 0 #None 

# General Parameters
NUM_BITS_PER_SYMBOL = 4
SYMBOL_RATE = 1e6                                               # [Hz], this defines the BW for the signal for both SC and OFDM
TOT_SYMBOLS_TO_DELIVER = 64 #  256*5*6+64+32 = 7776 for papr, 64 for BER

# Single Carrier Parameters
NUM_SC_SYMBOL = TOT_SYMBOLS_TO_DELIVER                                                                  # # number of symbol per SC block; to match # symbols as ofdm: (FFT_SIZE - CYCLIC_PRFX_LEN - 1) * (NUM_OFDM_SYMBOL - len(OFDM_SYMBOLS_FOR_PILOT_INDICES))
T_sc_symbol = 1/SYMBOL_RATE
T_sc_block = T_sc_symbol*NUM_SC_SYMBOL

# OFDM Parameters
FFT_SIZE = 32
CYCLIC_PRFX_LEN = 16
# SUBCARRIER_SPACING = int((SYMBOL_RATE/FFT_SIZE)/1e4)*1e4                                                    # [Hz], round the spacing to match papers results
SUBCARRIER_SPACING = int((SYMBOL_RATE/FFT_SIZE))                                                  # [Hz], round the spacing to match papers results
OFDM_SYMBOLS_FOR_PILOT_INDICES = [0]                                                                        # indices of the ofdm symbols that used as pilots for csi
NUM_OFDM_SYMBOL = int(TOT_SYMBOLS_TO_DELIVER/FFT_SIZE + len(OFDM_SYMBOLS_FOR_PILOT_INDICES))                      # number of OFDM symbol per block
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
