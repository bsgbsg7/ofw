#!/usr/bin/env python3
"""
Learned Q vs OFDM vs OTFS — 时变信道性能对比
==============================================
固定 Delay Spread = 620ns, 速度扫描 0 → 120 m/s, 纵轴 BER。

三种方案:
  1. Learned Q  — 网络学到的自适应 Q 矩阵
  2. OFDM       — 固定 IDFT 矩阵 (标准 OFDM)
  3. OTFS       — 理想 ISFFT/Heisenberg/SFFT 变换 (DD 域调制)

所有方案使用相同的:
  - 符号数: 64 (2 OFDM 符号 × 32 子载波)
  - 调制: 16-QAM (4 bits/symbol)
  - 信道: TDL-A, DS=620ns, 速度可变
  - CP 长度: 16
  - 无信道编码 (公平对比波形本身)
  - SNR: 15dB
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import *

N = FFT_SIZE          # 32 subcarriers
CP = CYCLIC_PRFX_LEN  # 16
M = 2                 # data OFDM symbols (excl. pilot)
N_SYM = N * M         # 64 total symbols

# ============================================================================
# 信道构建
# ============================================================================
def build_channel(speed):
    from utils.TDL_RandomDS import TDL_RandomDS
    ds = 620e-9
    return TDL_RandomDS(model='A',
        delay_spread_min=ds, delay_spread_max=ds + 1e-9,
        carrier_frequency=CARRIER_FREQ,
        min_speed=max(0.1, speed - 0.5), max_speed=speed + 0.5)

# ============================================================================
# 公共工具
# ============================================================================
def generate_symbols(batch_size):
    """生成随机 QAM 符号 (16-QAM, normalized)"""
    bits = np.random.randint(0, 2, (batch_size, N_SYM, 4))
    # 16-QAM mapping: Gray code
    real_part = (2 * bits[:, :, 0] - 1) * (1 + 2 * bits[:, :, 2])
    imag_part = (2 * bits[:, :, 1] - 1) * (1 + 2 * bits[:, :, 3])
    symbols = (real_part + 1j * imag_part).astype(np.complex64) / np.sqrt(10)
    return bits, symbols

def demap_symbols(rx_symbols):
    """16-QAM hard-decision demapping → bits. Inverts generate_symbols."""
    shape = rx_symbols.shape
    rx = rx_symbols * np.sqrt(10)  # undo normalization
    real = np.real(rx)  # values in [-3, -1, 1, 3]
    imag = np.imag(rx)

    # Decision: b0 = sign bit for real, b1 = sign bit for imag
    #           b2 = magnitude bit for real, b3 = magnitude bit for imag
    bits = np.zeros(shape + (4,), dtype=int)
    bits[..., 0] = (real > 0).astype(int)           # sign of real
    bits[..., 1] = (imag > 0).astype(int)           # sign of imag
    bits[..., 2] = (np.abs(real) > 2.0).astype(int) # magnitude of real
    bits[..., 3] = (np.abs(imag) > 2.0).astype(int) # magnitude of imag
    return bits

def compute_ber(tx_bits, rx_bits):
    return np.mean(tx_bits != rx_bits)

def run_channel(symbols_time, speed, snr_db):
    """
    通过 TDL 时变信道。
    symbols_time: (batch, total_time_samples) complex
    返回: (batch, total_time_samples) complex + noise
    """
    ch = build_channel(speed)
    batch_size = symbols_time.shape[0]
    n_time = symbols_time.shape[1]

    # 生成 CIR
    a, tau = ch(batch_size, n_time + 32 - 1, 1e6)  # bandwidth=1MHz
    # 手动做卷积
    l_tot = a.shape[-1]
    rx = np.zeros((batch_size, n_time), dtype=np.complex64)
    noise_power = 10 ** (-snr_db / 10)

    for b in range(batch_size):
        for l in range(l_tot):
            if np.abs(a[b, 0, 0, l]) > 1e-10:
                delay = int(tau[b, 0, 0, l] * 1e6)
                if delay < n_time:
                    contrib = a[b, 0, 0, l] * np.roll(symbols_time[b], delay)
                    contrib[:delay] = 0
                    rx[b] += contrib

    # AWGN
    rx_power = np.mean(np.abs(rx)**2)
    noise = np.sqrt(noise_power * rx_power / 2) * (
        np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))
    return rx + noise.astype(np.complex64)

def run_channel_sionna(model, x_time, speed, snr_db):
    """使用 Sionna 的时变信道 (更精确, 与训练一致)"""
    from sionna.phy.channel import cir_to_time_channel
    from sionna.phy.utils import ebnodb2no
    import sionna.phy as sn

    orig = model._channel_model
    ch = build_channel(speed)
    model._channel_model = ch

    batch_size = x_time.shape[0]
    no = ebnodb2no(snr_db, 4, 1, model._rg)

    a, tau = ch(batch_size, model._rg.num_time_samples + model._l_tot - 1,
                 model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                 l_min=model._l_min, l_max=model._l_max, normalize=True)

    x_time_tf = tf.constant(x_time, dtype=tf.complex64)
    # 补齐到 num_time_samples
    pad_len = model._rg.num_time_samples - x_time.shape[1]
    if pad_len > 0:
        x_time_tf = tf.pad(x_time_tf, [[0,0], [0, pad_len]])
    x_time_tf = tf.reshape(x_time_tf, [batch_size, 1, 1, 1, -1])
    y_time = model._channel_time(x_time_tf, h_time, no)
    y_time = y_time[..., -model._l_min:-model._l_max]
    y_time = y_time[:, 0, 0, 0, :x_time.shape[1]]

    model._channel_model = orig
    return y_time.numpy()

# ============================================================================
# Scheme 1: Learned Q
# ============================================================================
def learned_q_pipeline(model, bits, symbols, speed, snr_db):
    """
    使用网络学到的 Q + q 进行调制解调。
    """
    from sionna.phy.channel import cir_to_time_channel
    from sionna.phy.utils import ebnodb2no

    batch_size = symbols.shape[0]
    no = ebnodb2no(snr_db, 4, 1, model._rg)

    orig = model._channel_model
    ch = build_channel(speed)
    model._channel_model = ch

    # 生成信道
    a, tau = ch(batch_size, model._rg.num_time_samples + model._l_tot - 1,
                 model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                 l_min=model._l_min, l_max=model._l_max, normalize=True)

    # 提取 CIR 快照 → Q
    h_2d = h_time[:, 0, 0, 0, 0, :, :]
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    pilots = tf.gather(h_2d, indices, axis=1)
    pilots = tf.transpose(pilots, [0, 2, 1])
    Q, q_vec = model._qQ_creator_layer(pilots, training=False)

    # 构造频域输入: (batch, 1, 1, 3, 32) — [batch, num_tx, num_streams, num_symbols, fft_size]
    x_rg = np.zeros((batch_size, 1, 1, 3, N), dtype=np.complex64)
    # 填充 data symbols 到 symbol 0 和 2 (symbol 1 is pilot, skip)
    syms = symbols.reshape(batch_size, M, N)  # (batch, 2, 32)
    x_rg[:, 0, 0, 0, :] = syms[:, 0, :]   # data sym 0
    x_rg[:, 0, 0, 2, :] = syms[:, 1, :]   # data sym 1
    x_rg_tf = tf.constant(x_rg, dtype=tf.complex64)

    # Q 调制
    x_time = model._Q_modulator(Q, x_rg_tf)  # (batch, 1, 1, 1, 144)

    # 过信道
    y_time = model._channel_time(x_time, h_time, no)
    y_time = y_time[..., -model._l_min:-model._l_max]

    # Q 解调
    r_freq = model._Q_demodulator(Q, y_time)  # (batch, 1, 1, 1, 3, 32)

    # 提取 data symbols (skip pilot at idx 1)
    r_freq_3d = r_freq[:, 0, 0, :, :]  # (batch, 3, 32)
    r_data = tf.gather(r_freq_3d, [0, 2], axis=1)  # (batch, 2, 32)
    r_data_flat = tf.reshape(r_data, [batch_size, N_SYM])

    # q equalization: q_vec is (batch, N) → tile to (batch, N_SYM)
    q_tiled = tf.tile(q_vec, [1, M])  # repeat for each OFDM symbol
    r_eq = r_data_flat * q_tiled

    model._channel_model = orig
    return r_eq.numpy().reshape(batch_size, N_SYM)

# ============================================================================
# Scheme 2: OFDM (fixed IDFT)
# ============================================================================
def ofdm_pipeline(model, bits, symbols, speed, snr_db):
    """
    标准 OFDM: Q=IDFT, 使用 pilot (symbol 1) 做信道估计 + ZF 均衡。
    """
    from sionna.phy.channel import cir_to_time_channel
    from sionna.phy.utils import ebnodb2no

    batch_size = symbols.shape[0]
    no = ebnodb2no(snr_db, 4, 1, model._rg)

    orig = model._channel_model
    ch = build_channel(speed)
    model._channel_model = ch

    a, tau = ch(batch_size, model._rg.num_time_samples + model._l_tot - 1,
                 model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                 l_min=model._l_min, l_max=model._l_max, normalize=True)

    # 固定 IDFT
    idft_mat = np.exp(1j * 2 * np.pi * np.arange(N).reshape(-1,1) * np.arange(N) / N) / np.sqrt(N)
    Q_idft = tf.constant(idft_mat.astype(np.complex64))
    Q_tiled = tf.tile(Q_idft[None,:,:], [batch_size, 1, 1])

    # 提取 CIR → 获取 learned q 用于均衡 (公平对比: 相同的均衡器能力)
    h_2d = h_time[:, 0, 0, 0, 0, :, :]
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    pilots_cir = tf.gather(h_2d, indices, axis=1)
    pilots_cir = tf.transpose(pilots_cir, [0, 2, 1])
    _, q_vec = model._qQ_creator_layer(pilots_cir, training=False)

    # 构造频域输入: symbol 0,2=data, symbol 1=pilot (known as ones)
    x_rg = np.zeros((batch_size, 1, 1, 3, N), dtype=np.complex64)
    syms = symbols.reshape(batch_size, M, N)
    x_rg[:, 0, 0, 0, :] = syms[:, 0, :]
    x_rg[:, 0, 0, 1, :] = 1.0 + 0j  # pilot symbol = 1
    x_rg[:, 0, 0, 2, :] = syms[:, 1, :]
    x_rg_tf = tf.constant(x_rg, dtype=tf.complex64)

    x_time = model._Q_modulator(Q_tiled, x_rg_tf)
    y_time = model._channel_time(x_time, h_time, no)
    y_time = y_time[..., -model._l_min:-model._l_max]
    r_freq = model._Q_demodulator(Q_tiled, y_time)
    r_freq_3d = r_freq[:, 0, 0, :, :]  # (batch, 3, 32)

    # ---- 信道估计 & ZF 均衡 (用 pilot at symbol 1) ----
    pilot_rx = r_freq_3d[:, 1, :]  # (batch, 32), pilot_tx was all 1s
    # ZF: divide by estimated channel
    r_data = tf.gather(r_freq_3d, [0, 2], axis=1)  # (batch, 2, 32)
    h_est = pilot_rx  # since pilot_tx = 1
    r_eq = r_data / (h_est[:, tf.newaxis, :] + 1e-10)

    r_data_flat = tf.reshape(r_eq, [batch_size, N_SYM])

    model._channel_model = orig
    return r_data_flat.numpy().reshape(batch_size, N_SYM)

# ============================================================================
# Scheme 3: OTFS (增大的 DD 网格)
# ============================================================================

# OTFS 网格参数: 16 delay × 8 Doppler = 128 DD bins
# 其中 1 个 impulse pilot + guard band, 64 个 data (与 learned Q/OFDM 匹配)
N_DELAY   = 16
N_DOPPLER = 8
N_OTFS    = N_DELAY * N_DOPPLER                 # 128 DD bins
CP_OTFS   = 8                                    # CP 长度
N_TIME_SAMPLES_OTFS = N_DOPPLER * (N_DELAY + CP_OTFS)  # 8 × 24 = 192
# Pilot 位置和保护带
PILOT_DELAY   = 0
PILOT_DOPPLER = 0
GUARD_RADIUS  = 1  # 保护带半径 (DD bins), 确保足够 data bin

def build_dd_data_mask():
    """构建 DD 网格 mask: True = 数据位置, False = pilot/guard (含 Doppler 循环)"""
    mask = np.ones((N_DELAY, N_DOPPLER), dtype=bool)
    for d in range(N_DELAY):
        for dop in range(N_DOPPLER):
            # Doppler 循环距离
            dop_dist = min(abs(dop - PILOT_DOPPLER),
                          N_DOPPLER - abs(dop - PILOT_DOPPLER))
            dist = np.sqrt(float(d)**2 + float(dop_dist)**2)
            if dist <= GUARD_RADIUS:
                mask[d, dop] = False
    return mask

DD_DATA_MASK = build_dd_data_mask()
N_DATA_OTFS  = int(np.sum(DD_DATA_MASK))  # 可用数据 bin 数

def otfs_pipeline(model, bits, symbols, speed, snr_db):
    """
    OTFS 调制解调:
      DD grid: N_DELAY × N_DOPPLER = 16 × 8
      Impulse pilot + guard band → DD 信道估计 → MMSE 均衡
    """
    from sionna.phy.channel import cir_to_time_channel
    from sionna.phy.utils import ebnodb2no

    batch_size = symbols.shape[0]
    no = ebnodb2no(snr_db, 4, 1, model._rg)

    orig = model._channel_model
    ch = build_channel(speed)
    model._channel_model = ch

    a, tau = ch(batch_size, N_TIME_SAMPLES_OTFS + 64 - 1,
                 model._rg.bandwidth)
    h_time_full = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                     l_min=model._l_min, l_max=model._l_max, normalize=True)

    # ---- OTFS 调制: DD → TF → Time ----
    # Step 1: 构建 DD 网格
    X_DD = np.zeros((batch_size, N_DELAY, N_DOPPLER), dtype=np.complex64)
    # 放 pilot
    X_DD[:, PILOT_DELAY, PILOT_DOPPLER] = 1.0 + 0j
    # 放 data (只用到前 N_SYM=64 个 data 位置, 与 learned Q 使用相同符号数)
    data_positions = np.argwhere(DD_DATA_MASK)  # (N_DATA_OTFS, 2)
    n_data_use = min(N_SYM, N_DATA_OTFS)
    syms_flat = symbols.reshape(batch_size, N_SYM)[:, :n_data_use]
    for i in range(n_data_use):
        d, dop = data_positions[i]
        X_DD[:, d, dop] = syms_flat[:, i]

    # Step 2: ISFFT
    X_TF = np.fft.ifft(X_DD, n=N_DELAY, axis=1) * np.sqrt(N_DELAY)
    X_TF = np.fft.fft(X_TF, n=N_DOPPLER, axis=2) / np.sqrt(N_DOPPLER)
    # (batch, 16, 8) — freq × time

    # Step 3: Heisenberg (IDFT per column + CP)
    s_time = np.zeros((batch_size, N_TIME_SAMPLES_OTFS), dtype=np.complex64)
    SYM_LEN = N_DELAY + CP_OTFS  # 24
    for m in range(N_DOPPLER):
        col = X_TF[:, :, m]
        ts = np.fft.ifft(col, n=N_DELAY, axis=1) * np.sqrt(N_DELAY)
        with_cp = np.concatenate([ts[:, -CP_OTFS:], ts], axis=1)
        s_time[:, m*SYM_LEN:(m+1)*SYM_LEN] = with_cp

    # ---- 信道 (向量化时变卷积) ----
    l_tot = h_time_full.shape[-1]
    h_np = h_time_full[:, 0, 0, 0, 0, :, :].numpy()  # (batch, time, l_tot)
    # 向量化: y[t] = sum_l h[t,l] * s[t-l]
    T = N_TIME_SAMPLES_OTFS
    y_time = np.zeros((batch_size, T), dtype=np.complex64)
    for l in range(min(l_tot, T)):
        # h_np[:, :N_TIME_SAMPLES_OTFS, l] * shift s_time by l
        y_time[:, l:] += h_np[:, l:T, l] * s_time[:, :T-l]
    # AWGN
    noise_power = float(10 ** (-snr_db / 10))
    sig_power = np.mean(np.abs(y_time)**2, axis=1, keepdims=True)
    noise = np.sqrt(noise_power * sig_power / 2) * (
        np.random.randn(*y_time.shape) + 1j * np.random.randn(*y_time.shape))
    y_time += noise.astype(np.complex64)

    # ---- OTFS 解调: Time → TF → DD ----
    # Step 1: Wigner (remove CP + DFT)
    Y_TF = np.zeros((batch_size, N_DELAY, N_DOPPLER), dtype=np.complex64)
    for m in range(N_DOPPLER):
        start = m * SYM_LEN
        y_m = y_time[:, start+CP_OTFS:start+CP_OTFS+N_DELAY]
        Y_TF[:, :, m] = np.fft.fft(y_m, n=N_DELAY, axis=1) / np.sqrt(N_DELAY)

    # Step 2: SFFT = FFT_freq(IFFT_time) — inverse of ISFFT
    Y_DD = np.fft.fft(Y_TF, n=N_DELAY, axis=1) / np.sqrt(N_DELAY)      # FFT: freq→delay
    Y_DD = np.fft.ifft(Y_DD, n=N_DOPPLER, axis=2) * np.sqrt(N_DOPPLER)  # IFFT: time→Doppler

    # ---- DD 域 MPA 检测器 ----
    noise_var = np.float32(10 ** (-snr_db / 10))

    # Step 1: 从 guard 区域估计 DD 域信道冲激响应
    guard_mask = ~DD_DATA_MASK  # True = guard/pilot region
    guard_positions = np.argwhere(guard_mask)

    # 收集 guard 区域的信道响应 (所有 batch 平均, 得到稀疏的 DD CIR)
    h_dd_guard = np.zeros((N_DELAY, N_DOPPLER), dtype=np.complex128)
    for d, dop in guard_positions:
        h_dd_guard[d, dop] = np.mean(Y_DD[:, d, dop])

    # 找出显著的 DD 信道抽头 (幅度 > 阈值的)
    tap_threshold = 0.05 * np.max(np.abs(h_dd_guard))
    significant_taps = []
    for d in range(N_DELAY):
        for dop in range(N_DOPPLER):
            h_val = h_dd_guard[d, dop]
            if np.abs(h_val) > tap_threshold:
                significant_taps.append((d, dop, h_val))

    if not significant_taps:
        significant_taps = [(0, 0, h_dd_guard[0, 0])]

    # Step 2: MPA 迭代检测
    N_MPA_ITER = 8
    N_QAM = 16

    # 所有 DD 位置的符号估计 (先初始化为 0)
    X_est = np.zeros((batch_size, N_DELAY, N_DOPPLER), dtype=np.complex128)
    X_var = np.ones((batch_size, N_DELAY, N_DOPPLER))  # 符号方差

    # Pilot 位置已知
    X_est[:, PILOT_DELAY, PILOT_DOPPLER] = 1.0 + 0j
    X_var[:, PILOT_DELAY, PILOT_DOPPLER] = 0.0
    # Guard 位置为 0
    for d, dop in guard_positions:
        if d != PILOT_DELAY or dop != PILOT_DOPPLER:
            X_est[:, d, dop] = 0.0
            X_var[:, d, dop] = 0.0

    # 预处理: 对每个观测位置, 列出其影响的变量节点
    obs_to_var = {}  # (d,dop) → list of (tap_d, tap_dop, var_d, var_dop)
    for do in range(N_DELAY):
        for dopo in range(N_DOPPLER):
            if (do, dopo) in set(tuple(p) for p in guard_positions) and (do, dopo) != (PILOT_DELAY, PILOT_DOPPLER):
                continue  # guard 的观测不用于数据检测
            affected = []
            for td, tdop, _ in significant_taps:
                vd = (do - td) % N_DELAY
                vdop = (dopo - tdop) % N_DOPPLER
                affected.append((td, tdop, vd, vdop))
            obs_to_var[(do, dopo)] = affected
    # ---- 单抽头 MMSE 均衡 (DD 域) ----
    noise_var = np.float32(10 ** (-snr_db / 10))
    # DD 域等效信道增益 (从 pilot 位置估计)
    h_pilot = Y_DD[:, PILOT_DELAY, PILOT_DOPPLER]  # (batch,), pilot_tx = 1

    Y_DD_eq = np.zeros((batch_size, n_data_use), dtype=np.complex64)
    for i in range(n_data_use):
        d, dop = data_positions[i]
        y_val = Y_DD[:, d, dop]
        # MMSE: X_hat = conj(h) * y / (|h|^2 + sigma^2)
        h_abs2 = np.abs(h_pilot)**2 + noise_var
        Y_DD_eq[:, i] = y_val * np.conj(h_pilot) / np.maximum(h_abs2, 1e-10)

    model._channel_model = orig
    return Y_DD_eq


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("Learned Q vs OFDM vs OTFS — 时变信道性能对比")
    print(f"固定 Delay Spread = 620ns, SNR = 15dB")
    print("=" * 72)

    from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
    import sionna.phy as sn
    sn.config.seed = SEED

    # 加载模型
    with open('weights-qQ_Method_TV', 'rb') as f:
        weights = pickle.load(f)
    model = qQ_MODEL_TV(training=False)
    model(2, 20.0)
    model.set_weights(weights)
    print("模型加载完成。\n")

    # 速度扫描
    speeds = [0, 3, 10, 30, 60, 90, 120]
    snr_db = 15.0
    batch_size = 100
    n_trials = 5

    results = {
        'speeds': speeds,
        'learned': {'mean': [], 'std': []},
        'ofdm':    {'mean': [], 'std': []},
        'otfs':    {'mean': [], 'std': []},
    }

    print(f"{'Speed':>8s}  {'Learned Q':>14s}  {'OFDM':>14s}  {'OTFS':>14s}")
    print(f"{'':>8s}  {'BER':>6s} {'±std':>6s}  {'BER':>6s} {'±std':>6s}  {'BER':>6s} {'±std':>6s}")
    print("-" * 60)

    for speed in speeds:
        bers_learned = []
        bers_ofdm    = []
        bers_otfs    = []

        for trial in range(n_trials):
            bits, symbols = generate_symbols(batch_size)

            # Learned Q
            rx_learned = learned_q_pipeline(model, bits, symbols, speed, snr_db)
            rx_bits_learned = demap_symbols(rx_learned)
            bers_learned.append(compute_ber(bits, rx_bits_learned))

            # OFDM
            rx_ofdm = ofdm_pipeline(model, bits, symbols, speed, snr_db)
            rx_bits_ofdm = demap_symbols(rx_ofdm)
            bers_ofdm.append(compute_ber(bits, rx_bits_ofdm))

            # OTFS
            rx_otfs = otfs_pipeline(model, bits, symbols, speed, snr_db)
            rx_bits_otfs = demap_symbols(rx_otfs)
            bers_otfs.append(compute_ber(bits, rx_bits_otfs))

        results['learned']['mean'].append(np.mean(bers_learned))
        results['learned']['std'].append(np.std(bers_learned))
        results['ofdm']['mean'].append(np.mean(bers_ofdm))
        results['ofdm']['std'].append(np.std(bers_ofdm))
        results['otfs']['mean'].append(np.mean(bers_otfs))
        results['otfs']['std'].append(np.std(bers_otfs))

        print(f"  {speed:3.0f} m/s  {np.mean(bers_learned):.6f} ±{np.std(bers_learned):.4f}  "
              f"{np.mean(bers_ofdm):.6f} ±{np.std(bers_ofdm):.4f}  "
              f"{np.mean(bers_otfs):.6f} ±{np.std(bers_otfs):.4f}")

    # ========================================================================
    # 可视化
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # ---- 主图: BER vs Speed ----
    ax = axes[0]
    ax.errorbar(speeds, results['learned']['mean'], yerr=results['learned']['std'],
                color='#e41a1c', linewidth=2.5, marker='o', markersize=10,
                capsize=4, label='Learned Q', zorder=5)
    ax.errorbar(speeds, results['ofdm']['mean'], yerr=results['ofdm']['std'],
                color='#377eb8', linewidth=2.5, marker='s', markersize=10,
                capsize=4, label='OFDM (IDFT)', zorder=4)
    ax.errorbar(speeds, results['otfs']['mean'], yerr=results['otfs']['std'],
                color='#4daf4a', linewidth=2.5, marker='^', markersize=10,
                capsize=4, label='OTFS (ISFFT)', zorder=3)

    ax.set_xlabel('Speed [m/s]', fontsize=14)
    ax.set_ylabel('BER', fontsize=14)
    ax.set_title('Learned Q vs OFDM vs OTFS\n'
                 f'Delay Spread = 620ns, SNR = {snr_db:.0f}dB, 16-QAM, uncoded',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(bottom=0)

    # 标注
    ax.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
    ax.text(30, ax.get_ylim()[1]*0.95, '≈108 km/h', fontsize=8, color='orange', ha='center')
    ax.axvline(x=120, color='darkred', linestyle=':', alpha=0.5)
    ax.text(120, ax.get_ylim()[1]*0.95, '≈432 km/h', fontsize=8, color='darkred', ha='center')

    # ---- 增益图: Learned Q 相对于 OFDM / OTFS 的 BER 比率 ----
    ax = axes[1]
    x = np.arange(len(speeds))
    width = 0.35

    gain_vs_ofdm = [results['ofdm']['mean'][i] / (results['learned']['mean'][i] + 1e-10)
                    for i in range(len(speeds))]
    gain_vs_otfs = [results['otfs']['mean'][i] / (results['learned']['mean'][i] + 1e-10)
                    for i in range(len(speeds))]

    bars1 = ax.bar(x - width/2, gain_vs_ofdm, width, alpha=0.8, color='#377eb8',
                   label='OFDM / Learned Q')
    bars2 = ax.bar(x + width/2, gain_vs_otfs, width, alpha=0.8, color='#4daf4a',
                   label='OTFS / Learned Q')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Equal (1.0x)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in speeds])
    ax.set_xlabel('Speed [m/s]', fontsize=14)
    ax.set_ylabel('BER Ratio', fontsize=14)
    ax.set_title('Performance Gain: Learned Q vs Baselines\n'
                 '(> 1.0 = Learned Q is better)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, gain_vs_ofdm):
        color = 'green' if val > 1 else 'red'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=color)
    for bar, val in zip(bars2, gain_vs_otfs):
        color = 'green' if val > 1 else 'red'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=color)

    plt.suptitle(f'时变信道下三种波形方案对比 | Delay Spread = 620ns | 16-QAM uncoded',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('compare_learned_ofdm_otfs.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  保存: compare_learned_ofdm_otfs.png")

    # ---- 数据表格图 ----
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.axis('off')

    table_data = [['Speed', 'Learned Q BER', 'OFDM BER', 'OTFS BER',
                   'Learned vs OFDM', 'Learned vs OTFS']]
    for i, s in enumerate(speeds):
        table_data.append([
            f'{s} m/s',
            f'{results["learned"]["mean"][i]:.5f}',
            f'{results["ofdm"]["mean"][i]:.5f}',
            f'{results["otfs"]["mean"][i]:.5f}',
            f'{gain_vs_ofdm[i]:.2f}x',
            f'{gain_vs_otfs[i]:.2f}x',
        ])

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    # Header style
    for j in range(6):
        table[0, j].set_facecolor('#40466e')
        table[0, j].set_text_props(weight='bold', color='white')
    # Highlight best in each row
    for i in range(1, len(speeds)+1):
        bers = [results['learned']['mean'][i-1],
                results['ofdm']['mean'][i-1],
                results['otfs']['mean'][i-1]]
        best_idx = np.argmin(bers)
        table[i, best_idx+1].set_facecolor('#90ee90')

    ax2.set_title('BER Data Table | Delay Spread = 620ns, SNR = 15dB\n'
                  'Green = best scheme at each speed',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('compare_learned_ofdm_otfs_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  保存: compare_learned_ofdm_otfs_table.png")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 72)
    print("对比总结")
    print("=" * 72)

    print(f"\n  {'Speed':>8s}  {'Best Scheme':<16s}  {'Learned vs OFDM':<18s}  {'Learned vs OTFS':<18s}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*18}  {'-'*18}")
    for i, s in enumerate(speeds):
        bers = {
            'Learned Q': results['learned']['mean'][i],
            'OFDM': results['ofdm']['mean'][i],
            'OTFS': results['otfs']['mean'][i],
        }
        best = min(bers, key=bers.get)
        best_ber = bers[best]
        learned_ber = bers['Learned Q']
        print(f"  {s:3.0f} m/s  {best:<16s}  "
              f"Learned Q {'wins' if results['learned']['mean'][i] < results['ofdm']['mean'][i] else 'loses'} "
              f"({gain_vs_ofdm[i]:.2f}x)        "
              f"Learned Q {'wins' if results['learned']['mean'][i] < results['otfs']['mean'][i] else 'loses'} "
              f"({gain_vs_otfs[i]:.2f}x)")

    # 统计
    learned_wins_vs_ofdm = sum(1 for i in range(len(speeds))
                               if results['learned']['mean'][i] < results['ofdm']['mean'][i])
    learned_wins_vs_otfs = sum(1 for i in range(len(speeds))
                               if results['learned']['mean'][i] < results['otfs']['mean'][i])

    print(f"\n  Learned Q 优于 OFDM 的速度点数: {learned_wins_vs_ofdm}/{len(speeds)}")
    print(f"  Learned Q 优于 OTFS 的速度点数: {learned_wins_vs_otfs}/{len(speeds)}")

    print(f"\n  生成的文件:")
    print(f"    compare_learned_ofdm_otfs.png       — BER vs Speed 主图 + 增益对比")
    print(f"    compare_learned_ofdm_otfs_table.png — 数据表格")
    print(f"    analyze_q_doppler_mechanism.py      — 分析脚本")

if __name__ == "__main__":
    main()
