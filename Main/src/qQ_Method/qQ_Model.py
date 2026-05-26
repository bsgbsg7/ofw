"""
qQ_Method 端到端 OFDM 物理层模型

核心思想：用神经网络学习一个 Q 矩阵，替代传统 OFDM 中的 IFFT/FFT 调制解调，
        在保持 BER 性能的同时降低 PAPR（峰均比）。

信号流：
  比特 → QAM映射 → 资源网格 → Q调制(替换IFFT) → 时域信道 → Q解调(替换FFT) → 均衡 → 解映射 → LLR

损失函数：
  total_loss = w_bce * BCE_loss + w_papr * PAPR_loss
  其中权重 w_bce, w_papr 由 Uncertainty 网络根据信道 RMS 延迟扩展自适应学习
"""

import tensorflow as tf
import keras
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import *
from channel import channel_model
import numpy as np
from sionna.phy import Block
from sionna.phy.mimo.stream_management import StreamManagement
from sionna.phy.ofdm.resource_grid import ResourceGrid
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.channel import  time_lag_discrete_time_channel, OFDMChannel, ApplyTimeChannel, cir_to_time_channel,cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber, flatten_last_dims, hard_decisions
import sionna.phy as sn
from src.qQ_Method.qQ_creator_layer import qQ_creator_layer, OrtQ_creator_layer, qQ_creator_conv_gru
from src.qQ_Method.Q_Modulator import Q_Modulator
from src.qQ_Method.Q_Demodulator import Q_Demodulator
from src.qQ_Method.qQ_uncertainty_model import UncertaintyModel_1D, UncertaintyModel_2D
from utils.PAPR import emprical_papr
from utils.General_helpers import make_shift_P
import matplotlib.pyplot as plt

class qQ_MODEL(keras.Model):
    """
    Q-Modulation 端到端模型

    继承 keras.Model，实现自定义的物理层前向传播。
    训练模式返回 (total_loss, PAR, bce_loss)，评估模式返回 (发送比特, 接收比特)。

    关键子模块：
      - _qQ_creator_layer:   神经网络，根据信道冲激响应生成 Q 矩阵和 q 向量
      - _Q_modulator:        用 Q 矩阵替代 IFFT，将频域符号变换到时域
      - _Q_demodulator:      用 Q 矩阵的共轭转置替代 FFT，将时域信号变换回频域
      - _UncertaintyModel:   根据 RMS 延迟扩展预测多任务损失的权重
    """

    def __init__(self,
                 training=False,
                 visulaize=False,
                 BS_ant=1,
                 UT_ant=1):
        super().__init__()

        # ========== 通用设置 ==========
        self.CCDF_mode = False                    # CCDF 模式：仅输出时域信号用于 PAPR 统计分析
        self.visulaize_progress = visulaize       # 若 True，移除 tf.function 加速以便调试可视化

        # ========== 系统参数（从 config.py 导入） ==========
        self._tot_symbol_to_deliver = TOT_SYMBOLS_TO_DELIVER     # 每帧要传输的 QAM 符号总数
        self._carrier_frequency = CARRIER_FREQ                    # 载波频率 (Hz)
        self._subcarrier_spacing = SUBCARRIER_SPACING             # 子载波间隔 (Hz)
        self._fft_size = FFT_SIZE                                 # FFT 大小（也是子载波数）
        self._cyclic_prefix_length = CYCLIC_PRFX_LEN              # 循环前缀长度
        self._num_ofdm_symbols = NUM_OFDM_SYMBOL                  # 每帧 OFDM 符号数
        self._num_ut_ant = UT_ant                                  # 用户天线数
        self._num_bs_ant = BS_ant                                  # 基站天线数
        self._num_streams_per_tx = self._num_ut_ant               # 每发射天线的数据流数
        self._dc_null = False                                     # 是否置空 DC 子载波
        self._num_guard_carriers = [0, 0]                         # 保护带子载波数
        self._pilot_pattern = "kronecker"                         # 导频图案类型
        self._pilot_ofdm_symbol_indices = OFDM_SYMBOLS_FOR_PILOT_INDICES  # 导频所在的 OFDM 符号索引
        self._num_bits_per_symbol = NUM_BITS_PER_SYMBOL           # 每符号比特数（4 → 16QAM）
        self._coderate = 1                                        # 编码率（1 = 无编码）

        # ========== 系统组件 ==========

        # 流管理：定义发射/接收天线之间的数据流映射
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        # 资源网格：定义 OFDM 时频资源结构
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        # 每帧传输的总比特数 = 数据子载波数 × 每符号比特数
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)

        # ========== 信道模型 ==========
        self._channel_model = channel_model     # 从 channel.py 导入（TDL 随机延迟扩展信道）
        self._awgn_channel = sn.channel.AWGN()  # AWGN 信道（用于单独测试）

        # 时域信道相关参数
        l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_min = L_MIN if L_MIN is not None else l_min        # 最小时延抽头索引
        self._l_tot = self._l_max - self._l_min + 1                # 多径抽头总数

        # 时域信道卷积层：实现 y[t] = sum(h[l] * x[t-l]) + noise
        self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                l_tot=self._l_tot,
                                                add_awgn=True)

        # 频域 OFDM 信道层（用于对比/调试）
        self._channel_freq = OFDMChannel(self._channel_model, self._rg,
                                         add_awgn=True, normalize_channel=True, return_channel=True)

        # ========== 发射链路组件 ==========
        self._binary_source = BinarySource()    # 随机比特生成器
        self._mapper = Mapper("pam" if self._num_bits_per_symbol == 1 else 'qam',
                              self._num_bits_per_symbol)      # 比特 → QAM 符号
        self._rg_mapper = ResourceGridMapper(self._rg)        # QAM 符号 → 资源网格映射

        # 传统 OFDM 调制/解调器（用于对比和 CSI 生成时的备选方案）
        self.OFDM_modulator = OFDMModulator(self._cyclic_prefix_length)
        self.OFDM_demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        # ========== 接收链路组件 ==========
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")  # LS 信道估计
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)                 # LMMSE 均衡
        self._demapper = Demapper("app",
                                  "pam" if self._num_bits_per_symbol == 1 else 'qam',
                                  self._num_bits_per_symbol,
                                  hard_out=False)                              # QAM 符号 → LLR (软解调)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)           # 移除空子载波

        # ========== 训练参数 ==========
        self.training = training
        self.Q_as_ifft = False                               # 是否直接用 IFFT 矩阵作为 Q（消融实验）
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')  # BCE 损失（from_logits=True 表示输入是 LLR）
        self._epsilon_P = 7                                  # PAPR 相关的阈值参数

        # ========== 神经网络层 ==========

        # Q 矩阵生成网络（核心）：输入信道冲激响应，输出 Q 矩阵和 q 向量
        # 可选方案：
        #   1) qQ_creator_layer — 纯 Dense 网络
        #   2) qQ_creator_transformer_layer — Transformer 架构
        #   3) qQ_creator_conv_gru — Conv1D + GRU 架构（当前使用）
        self._qQ_creator_layer = qQ_creator_conv_gru(self._fft_size)

        # Q 调制解调器：用 Q 矩阵替代传统 IFFT/FFT
        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        # 不确定性网络：根据 RMS 延迟扩展预测各损失项的权重
        # UncertaintyModel_2D: 输出 log_sigma_par, log_sigma_bce, log_sigma_par_lim
        # UncertaintyModel_1D: 输出 PAPR 约束阈值 par_lim
        self._UncertaintyModel_bce_par = UncertaintyModel_2D()
        self._UncertaintyModel_par_lim = UncertaintyModel_1D()

    # @tf.function
    def call(self, batch_size, ebno_db):
        """
        前向传播（训练或评估）

        数据流（完整管线）：
        1. 比特生成:    随机生成 batch_size 组比特
        2. QAM 映射:    比特 → QAM 星座符号
        3. 资源网格:    QAM 符号放置到 OFDM 时频网格
        4. 信道估计:    发送 delta 脉冲获取时域信道冲激响应 (CIR)
        5. Q 矩阵生成:  神经网络根据 CIR 生成调制矩阵 Q 和均衡向量 q
        6. Q 调制:      用 Q 矩阵替代 IFFT，将频域符号转为时域信号
        7. 时域信道:    时域信号经过多径信道 + AWGN
        8. Q 解调:      用 Q^H 替代 FFT，将接收时域信号转回频域
        9. 均衡 & 解映射: 用 q 向量做频域均衡，然后 QAM 解映射得到 LLR
        10. 损失计算:    多任务损失 = BCE(通信) + PAPR(峰均比)，权重由 Uncertainty 网络自适应

        Args:
            batch_size: 批次大小
            ebno_db:    Eb/N0 (dB)，信噪比

        Returns:
            训练模式: (total_loss, par_mean, bce_mean)
            评估模式: (b, b_hat) — 发送比特和硬判决后的接收比特
        """

        # -------- 步骤 1-3: 比特生成、QAM 映射、资源网格 --------
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])  # 随机比特
        x = self._mapper(b)                  # 比特 → QAM符号
        x_rg = self._rg_mapper(x)            # 映射到资源网格 [batch, 1, 1, num_ofdm_symbols, fft_size]

        # -------- 步骤 4: 信道估计（获取 CSI） --------
        # 4a. 生成时域信道冲激响应
        a, tau = self._channel_model(batch_size,
                                     self._rg.num_time_samples+self._l_tot-1,
                                     self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                      l_min=self._l_min, l_max=self._l_max, normalize=True)

        # 4b. 生成频域信道响应（用于可视化对比）
        a_freq = a[..., self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
        a_freq = a_freq[..., :self._rg.num_ofdm_symbols]
        h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
        h_freq = self._remove_nulled_scs(h_freq)

        # -------- 步骤 5: CSI 获取与 Q 矩阵生成 --------
        # 方案 A (False): 通过完整 OFDM 链路获取 CSI（较慢，用于调试）
        if False:
            x_time_for_csi = self.OFDM_modulator(x_rg)
            y_time = self._channel_time(x_time_for_csi, h_time, 0)
            y_time = y_time[...,-0:-self._l_max]
            y = self.OFDM_demodulator(y_time)
            h, err_var = self._ls_est(y, no)
            x_hat_debug, no_eff = self._lmmse_equ(y, h, err_var, no)
            llr = self._demapper(x_hat_debug, no_eff)
            b_hat_debug = hard_decisions(llr)
            channel_freq_domain = h[:,0,0,0,0,0,:]

        # 方案 B (True): 直接发送 delta 脉冲获取时域 CIR（高效）
        if True:
            # 构造 delta 脉冲：在频域第一个子载波上为 1 + 0j，其余为 0
            x_time_for_csi = x_rg[:,:,:,0,:]
            delta = tf.tile(
                tf.reshape(tf.complex(tf.one_hot(0, self._fft_size, dtype=tf.float32),
                                       tf.zeros(self._fft_size, dtype=tf.float32)),
                           [1,1,1,self._fft_size]),
                [tf.shape(x_rg)[0],1,1,1])

            # 将 delta 脉冲补零到与 OFDM 调制后等长，以便通过时域信道
            diff = tf.shape(self.OFDM_modulator(x_rg))[-1] - tf.shape(x_time_for_csi)[-1]
            paddings = tf.stack([[0, 0], [0, 0], [0, 0], tf.stack([0, diff])])
            delta_padded = tf.pad(delta, paddings, mode='CONSTANT', constant_values=tf.complex(0.0, 0.0))

            # delta 脉冲通过时域信道 → 直接得到信道冲激响应
            y_time = self._channel_time(delta_padded, h_time, 0)
            y_time = y_time[...,-self._l_min:-self._l_max]

            # 提取信道导频响应（即 CIR 的抽头值）
            pilots_post_channel = tf.expand_dims(y_time[:,0,0,:self._l_max], axis=-1)

            # 计算 RMS 延迟扩展 (Root Mean Square Delay Spread)
            # DS_rms = sqrt( E[tau^2] - E[tau]^2 )，用于表征多径信道的频率选择性
            h = tf.squeeze(pilots_post_channel, axis=-1)   # (batch, 抽头数)
            delays = tf.range(tf.shape(h)[1], dtype=tf.float32)
            power = tf.square(tf.abs(h))
            mean_delay = tf.reduce_sum(delays * power, axis=-1) / tf.reduce_sum(power, axis=-1)
            rms_ds = tf.sqrt(
                tf.reduce_sum(power * tf.square(delays - mean_delay[:, None]), axis=-1)
                / tf.reduce_sum(power, axis=-1)
            )
            rms_ds = tf.expand_dims(rms_ds, -1)

        # 神经网络根据 CIR 生成 Q 矩阵 (N×N) 和 q 均衡向量 (N 维)
        Q, q = self._qQ_creator_layer(pilots_post_channel, training=self.training)

        # -------- 步骤 6: Q 调制（替代 IFFT） --------
        # x_freq [batch, 1, 1, num_symbols, fft_size] @ Q[batch, N, N]
        # → x_time [batch, 1, 1, time_samples]
        x_time = self._Q_modulator(Q, x_rg)

        # CCDF 模式：仅返回时域信号用于 PAPR 统计
        if self.CCDF_mode:
            return x_time[:,0,0,:], rms_ds

        # -------- 步骤 7: 时域信道 --------
        # 多径卷积 + AWGN
        y_time = self._channel_time(x_time, h_time, no)
        y_time = y_time[...,-self._l_min:-self._l_max]

        # -------- 步骤 8: Q 解调（替代 FFT） --------
        # 用 Q^H 将时域接收信号转回频域
        r_freq = self._Q_demodulator(Q, y_time)

        # -------- 步骤 9: 频域均衡 & 解映射 --------
        # 移除第一个 OFDM 符号（可能被用于导频）
        r_freq = r_freq[:,:,:,1:,:]

        # q 向量广播：对每个子载波做逐元素均衡（类似单抽头均衡）
        q = q[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        q = tf.tile(q, [1, 1, 1, tf.shape(r_freq)[-2], 1])
        r_freq_equalzied = r_freq * q

        # 展平频域符号以匹配 QAM 符号格式
        current_shape = tf.shape(r_freq_equalzied)
        r_freq_equalzied = tf.reshape(r_freq_equalzied,
                                      [current_shape[0], current_shape[1], current_shape[2], -1])
        r_freq_equalzied.set_shape([None, None, None, self._tot_symbol_to_deliver])

        # 软解调：均衡后的 QAM 符号 → LLR (Log-Likelihood Ratio)
        llr = self._demapper(r_freq_equalzied, no)

        # -------- 步骤 10: 损失计算（仅训练模式） --------
        # 不确定性网络：根据 RMS 延迟扩展预测各损失的 log-方差权重
        # 原理：不同信道条件下，BER 和 PAPR 的重要性不同
        #   - 延迟扩展大 → 频率选择性严重 → BER 权重应该更大
        #   - 延迟扩展小 → 频率平坦 → 可以更多关注 PAPR

        log_sigma_par, log_sigma_bce, log_sigma_par_lim = \
            self._UncertaintyModel_bce_par(rms_ds, training=self.training)
        par_lim = self._UncertaintyModel_par_lim(rms_ds, training=self.training)

        if self.training:
            # BCE 损失：衡量比特传输质量（通信性能）
            bce_loss = tf.squeeze(self.bce(tf.reshape(b, tf.shape(llr)), llr))

            # PAPR 损失：衡量时域信号峰均比
            PAR = emprical_papr(tf.squeeze(x_time), None, par_lim)

            # 总损失 = 自适应加权的多任务损失
            # 使用 Gaussian 不确定性建模：
            #   loss = exp(-log_sigma) * task_loss + log_sigma
            # exp(-log_sigma) 即 task_weight，log_sigma 是正则项防止权重退化为 0
            total_loss = tf.reduce_mean(
                tf.exp(-log_sigma_bce) * bce_loss +
                tf.exp(-log_sigma_par) * PAR +
                tf.exp(-log_sigma_par_lim) * par_lim +
                log_sigma_bce + log_sigma_par + log_sigma_par_lim
            )

            par_mean = tf.reduce_mean(PAR)
            bce_mean = tf.reduce_mean(bce_loss)

            # 可视化（可选，仅在 visulaize_progress=True 时启用）
            if self.visulaize_progress:
                self.visulaize(h_freq, Q, rms_ds,
                              tf.exp(-log_sigma_bce), tf.exp(-log_sigma_par))

            return total_loss, par_mean, bce_mean

        else:
            # 评估模式：硬判决后返回发送/接收比特用于计算 BER
            b_hat = hard_decisions(llr)
            b_hat = tf.reshape(b_hat, tf.shape(b))
            return b, b_hat

    def visulaize(self, h_freq, Q, ds_rms=None, w_bce=None, w_papr=None):
        """
        可视化函数：绘制频域信道响应、Q 矩阵时频特性、DS 与损失权重关系

        生成以下图表：
          - 最小/最大延迟扩展样本的频域信道响应
          - 最小/最大延迟扩展样本的 Q 矩阵行向量（时域波形）
          - 最小/最大延迟扩展样本的 Q 矩阵行向量（频谱）
          - RMS 延迟扩展 vs BCE/PAPR 权重散点图
        """

        ds_rms_np = ds_rms.numpy() if hasattr(ds_rms, "numpy") else ds_rms

        # 找到 batch 中延迟扩展最小和最大的样本索引
        idx_min = np.argmin(ds_rms_np)
        idx_max = np.argmax(ds_rms_np)

        Q_min = Q[idx_min,:,:]
        Q_max = Q[idx_max,:,:]
        h_freq_min = h_freq[idx_min,0,0,0,0,0,:]
        h_freq_max = h_freq[idx_max,0,0,0,0,0,:]
        N = self._fft_size
        how_many_waves_to_plots = 8   # 可视化的 Q 矩阵行数

        # ---- 频域信道响应（最小 DS 样本） ----
        plt.figure()
        plt.plot(np.abs(h_freq_min))
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.ylim(0,2)
        plt.title(f"Channel frequency responses - DelaySpread in RMS: {ds_rms_np[idx_min]}")
        plt.savefig('_ofdm_model_csi_estimation_min_DS.png')
        plt.close()

        # ---- 频域信道响应（最大 DS 样本） ----
        plt.figure()
        plt.plot(np.abs(h_freq_max))
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.ylim(0,2)
        plt.title(f"Channel frequency responses - DelaySpread in RMS: {ds_rms_np[idx_max]}")
        plt.savefig('_ofdm_model_csi_estimation_max_DS.png')
        plt.close()

        # ---- Q 矩阵行向量时域可视化（最小 DS 样本） ----
        fig, axes = plt.subplots(how_many_waves_to_plots, 1, figsize=(12, 2*how_many_waves_to_plots), sharex=True)
        for i in range(how_many_waves_to_plots):
            row = Q_min[i, :]
            axes[i].plot(np.real(row), label="Real", color="b")
            axes[i].plot(np.imag(row), label="Imag", color="r", linestyle="--")
            axes[i].set_title(rf"Row {i} of Q Matrix - $f_{i}[n]$")
            axes[i].set_ylabel(rf"$f_{i}[n]$")
            axes[i].set_xlabel(rf"n")
            axes[i].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('_qQ_functions_Time_Visualization_min_DS.png')
        plt.close()

        # ---- Q 矩阵行向量时域可视化（最大 DS 样本） ----
        fig, axes = plt.subplots(how_many_waves_to_plots, 1, figsize=(12, 2*how_many_waves_to_plots), sharex=True)
        for i in range(how_many_waves_to_plots):
            row = Q_max[i, :]
            axes[i].plot(np.real(row), label="Real", color="b")
            axes[i].plot(np.imag(row), label="Imag", color="r", linestyle="--")
            axes[i].set_title(rf"Row {i} of Q Matrix - $f_{i}[n]$")
            axes[i].set_ylabel(rf"$f_{i}[n]$")
            axes[i].set_xlabel(rf"n")
            axes[i].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('_qQ_functions_Time_Visualization_max_DS.png')
        plt.close()

        # ---- Q 矩阵行向量频谱可视化（最小 DS 样本） ----
        M = 2048 * N   # 大 FFT 点数以获得更精细的频谱分辨率
        freqs = np.fft.fftfreq(M, d=1.0)
        freqs = np.fft.fftshift(freqs)
        plt.figure(figsize=(10, 6))
        for i in range(how_many_waves_to_plots):
            row = Q_min[i, :]
            spectrum = np.fft.fft(row, n=M)
            spectrum = np.fft.fftshift(spectrum)
            plt.plot(freqs, np.abs(spectrum), label=f"Row {i}")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude (linear)")
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.savefig('_qQ_functions_Freq_Visualization_min_DS.png')
        plt.close()

        # ---- Q 矩阵行向量频谱可视化（最大 DS 样本） ----
        M = 2048 * N
        freqs = np.fft.fftfreq(M, d=1.0)
        freqs = np.fft.fftshift(freqs)
        plt.figure(figsize=(10, 6))
        for i in range(how_many_waves_to_plots):
            row = Q_max[i, :]
            spectrum = np.fft.fft(row, n=M)
            spectrum = np.fft.fftshift(spectrum)
            plt.plot(freqs, np.abs(spectrum), label=f"Row {i}")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude (linear)")
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.savefig('_qQ_functions_Freq_Visualization_max_DS.png')
        plt.close()

        # ---- RMS 延迟扩展 vs 损失权重散点图 ----
        if ds_rms is not None and w_bce is not None and w_papr is not None:
            ds_rms_np = ds_rms.numpy() if hasattr(ds_rms, "numpy") else ds_rms
            w_bce_np = w_bce.numpy() if hasattr(w_bce, "numpy") else w_bce
            w_papr_np = w_papr.numpy() if hasattr(w_papr, "numpy") else w_papr

            # BCE 权重 vs DS
            plt.figure(figsize=(8,6))
            plt.scatter(ds_rms_np, w_bce_np, color='b', label='BCE weight', alpha=0.7)
            plt.xlabel("RMS Delay Spread")
            plt.ylabel("BCE Loss weight")
            plt.title("Per-sample Uncertainty BCE Weights vs RMS Delay Spread")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('_ds_vs_BCE_weights.png')
            plt.close()

            # PAPR 权重 vs DS
            plt.figure(figsize=(8,6))
            plt.scatter(ds_rms_np, w_papr_np, color='r', label='PAPR weight', alpha=0.7)
            plt.xlabel("RMS Delay Spread")
            plt.ylabel("PAPR Loss weight")
            plt.title("Per-sample Uncertainty PAPR Weight vs RMS Delay Spread")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('_ds_vs_PAPR_weights.png')
            plt.close()

        pass

    def training_log(self, total_loss, bce_loss, PAR, llr, bits, orthogonalty_loss=tf.constant([0])):
        """
        训练日志打印函数：计算并打印当前 BER、各损失项

        Args:
            total_loss:        总损失
            bce_loss:          BCE 损失
            PAR:               PAPR 损失
            llr:               软解调输出的 LLR
            bits:              原始发送比特
            orthogonalty_loss: 正交性损失（预留，当前未使用）
        """
        b_hat = hard_decisions(llr)
        b_hat = tf.reshape(b_hat, tf.shape(bits))
        ber = compute_ber(bits, b_hat)
        tf.print("Total Loss:", total_loss,
                    " | BCE:", bce_loss,
                    " | PAR:", PAR,
                    " | BER:", ber)

        return {'Total_Loss': total_loss, 'bce_Loss':bce_loss, 'BER':ber}

if __name__ == "__main__":
    # 快速测试：创建模型并跑一次前向传播
    sn.config.seed = SEED
    model = qQ_MODEL(training=True, visulaize=True)
    model(100,40)   # batch_size=100, Eb/N0=40dB
