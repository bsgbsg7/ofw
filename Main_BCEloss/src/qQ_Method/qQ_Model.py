"""
qQ_Method 端到端 OFDM 物理层模型 (BCE-only 版本)

核心思想：用神经网络学习一个 Q 矩阵，替代传统 OFDM 中的 IFFT/FFT 调制解调，
        在保持 BER 性能的同时降低 PAPR（峰均比）。

信号流：
  比特 → QAM映射 → 资源网格 → Q调制(替换IFFT) → 时域信道 → Q解调(替换FFT) → 均衡 → 解映射 → LLR

损失函数：
  total_loss = mean(BCE_loss)   （仅使用二进制交叉熵损失，不使用 PAPR 损失）
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
        # ebnodb2no: Eb/N0 (dB) → 噪声方差, 根据调制阶数/码率/资源网格计算每符号噪声功率
        # 将 Eb/N0 (dB) 转换为噪声方差，根据调制阶数、码率、资源网格计算出每符号的噪声功率
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        # BinarySource: 随机比特生成器, shape=[batch, 1, num_streams, num_bits]
        # 批量大小 × 1(单发射端) × 流数 × 每帧总比特数
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        # 作用：将二进制比特调制为 QAM 复数星座点。】
        # 输入: [1, 0, 1, 1]  (4个比特)
        # 输出: -0.316 + 0.948j  (一个复数，对应16QAM星座图上一个点)
        # 即把 [batch, 1, num_streams, num_bits] 的比特流 → [batch, 1, num_streams, num_symbols] 的复数符号序列。
        # Mapper 决定"一个符号长什么样"
        x = self._mapper(b)       
        # 作用：将扁平的 QAM 符号序列填充到 OFDM 时频资源网格的正确位置上，同时插入导频。
        # 即把 [batch, 1, num_streams, num_symbols] → [batch, 1, num_streams, num_ofdm_symbols, fft_size] 
        # ResourceGridMapper 决定"符号摆在哪里"。   
        x_rg = self._rg_mapper(x)            # 映射到资源网格 [batch, 1, 1, num_ofdm_symbols, fft_size]

        # -------- 步骤 4: 信道估计（获取 CSI） --------
        # 4a. 生成时域信道冲激响应
        # a   — 每条多径的复增益 (amplitude)
        # tau — 每条多径的时延 (delay, 单位: 秒)
        a, tau = self._channel_model(batch_size,
                                     self._rg.num_time_samples+self._l_tot-1,
                                     self._rg.bandwidth)
        # 每个抽头 h[l] 存储的是：所有时延落在第 l 个采样间隔内的多径的叠加
        # 核心转换：连续时延 tau 按采样间隔量化到离散抽头 l。l_min/l_max 指定了抽头索引范围，bandwidth 决定采样间隔。
        # 物理含义：h_time[l] 就是多径信道在第 l 个抽头的复增益，
        # 接收信号 = sum(h_time[l] * x[t-l]) + noise，对应后面 ApplyTimeChannel 做的时域卷积
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                      l_min=self._l_min, l_max=self._l_max, normalize=True)

        # 4b. 生成频域信道响应（用于可视化对比）
        # 从时变幅度 a 中按 OFDM 符号周期抽取一个代表值:
        #   步长 = fft_size + cp_len (一个完整OFDM符号的采样数)
        #   起始位置 = cp_len (跳过第一个CP，取有用部分的第一个采样)
        #   效果: a 从逐采样点 → 逐OFDM符号, shape [..., num_ofdm_symbols]
        a_freq = a[..., self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
        a_freq = a_freq[..., :self._rg.num_ofdm_symbols]  # 截取恰好 num_ofdm_symbols 个符号
        # cir_to_ofdm_channel: 用时延 tau + 每符号幅度 a_freq 计算每个子载波频率上的频域信道响应
        #   h_freq[f] = sum_k a_freq[k] * exp(-j*2π * f * tau[k])
        h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
        # 移除保护带/DC空子载波, 只保留数据子载波上的信道系数
        h_freq = self._remove_nulled_scs(h_freq)
        # 上面的 h_freq 是为了看图，下面的求解是为了得到papr权重

        # -------- 步骤 5: 获取 CIR 并计算 RMS 延迟扩展 --------
        # 目的: 得到 RMS 延迟扩展 rms_ds, 供 Uncertainty 网络自适应调节损失权重
            # 信道差（RMS_DS 大）→ 降低 PAPR loss 权重，优先保 BER
            # 信道好（RMS_DS 小）→ 提高 PAPR loss 权重，趁机压低峰均比
        # 两种方案获取 CIR (pilots_post_channel):
        #   A) 完整 OFDM 链路: 调制→信道→解调→LS估计 (慢, 模拟真实接收)
        #   B) delta 脉冲探测: 发单点脉冲过信道, 输出即是 CIR (快, 等效但省去调制解调)

        # 方案 A (False): 完整 OFDM 链路 — 导频→信道→LS信道估计→频域CSI
        if False:
            x_time_for_csi = self.OFDM_modulator(x_rg)          # IFFT + 加CP
            y_time = self._channel_time(x_time_for_csi, h_time, 0)  # 通过时域信道
            y_time = y_time[...,-0:-self._l_max]
            y = self.OFDM_demodulator(y_time)                   # 去CP + FFT
            h, err_var = self._ls_est(y, no)                    # LS 信道估计
            x_hat_debug, no_eff = self._lmmse_equ(y, h, err_var, no)  # LMMSE 均衡
            llr = self._demapper(x_hat_debug, no_eff)           # 软解调 → LLR
            b_hat_debug = hard_decisions(llr)                   # LLR → 硬判比特
            channel_freq_domain = h[:,0,0,0,0,0,:]

        # 方案 B (True): delta 脉冲探测 — 频谱平坦, 输出 = CIR 本身 (y = δ * h = h)
        if True:
            # 频域 delta: 只在 DC 子载波放 1+0j, 其余置零 → 平坦频谱
            x_time_for_csi = x_rg[:,:,:,0,:]    # 资源网格 [batch, 1, 1, num_ofdm_symbols, fft_size]
            delta = tf.tile(
                tf.reshape(tf.complex(tf.one_hot(0, self._fft_size, dtype=tf.float32),
                                       tf.zeros(self._fft_size, dtype=tf.float32)),
                           [1,1,1,self._fft_size]),
                [tf.shape(x_rg)[0],1,1,1])

            # 补零对齐: delta 脉冲补零到 OFDM 调制后同等长度, 保证时域卷积维度匹配
            diff = tf.shape(self.OFDM_modulator(x_rg))[-1] - tf.shape(x_time_for_csi)[-1]
            paddings = tf.stack([[0, 0], [0, 0], [0, 0], tf.stack([0, diff])])
            delta_padded = tf.pad(delta, paddings, mode='CONSTANT', constant_values=tf.complex(0.0, 0.0))

            # delta 脉冲通过时域信道 → 输出即信道冲激响应 (CIR)
            y_time = self._channel_time(delta_padded, h_time, 0)
            # 经过时域卷积 ApplyTimeChannel 后，y_time 的总长度 = 发射信号长度 + 多径抽头数 - 1 （尾部多出了多径拖尾）。
            # 为了后续 OFDM 解调（FFT），需要把接收信号截回到正确长度。
                # -self._l_min — 起始位置（从末尾倒推 l_min 个样本，l_min 通常为 0，即从头开始）
                # -self._l_max — 终止位置（去掉末尾 l_max 个样本的多径拖尾）
            # 物理直觉：假设 l_max = 5，发射了 80 个采样，卷积后收端得到 84 个采样（多了 4 个多径拖尾），这一步 [...: -5] 就是切掉末尾 5 个样本，恢复正确长度以供 FFT 解调。
            # 0 是起始索引，-5 是终止索引（倒数第 5 个），所以结果是 95 个样本，只是切掉了末尾 5 个多径拖尾
            y_time = y_time[...,-self._l_min:-self._l_max]

            # 提取 CIR: y_time[batch, tx, rx, taps] → [batch, l_max, 1]
            #   [:, 0, 0, :l_max] 取第1个收发对的全部batch、前l_max个抽头
            #   expand_dims(axis=-1) 补通道维, 适配下游 _qQ_creator_layer 的输入 shape
            #           shape 从 [batch, l_max] → [batch, l_max, 1]
            pilots_post_channel = tf.expand_dims(y_time[:,0,0,:self._l_max], axis=-1)

            # delta脉冲过信道 → y_time 原始长度 = 脉冲长度 + l_tot - 1

            # 第1次: y_time[..., -l_min : -l_max]      ← 去尾: 切掉卷积拖尾的多余样本
            # 第2次: y_time[:, 0, 0, :l_max]           ← 做两件事:
            #                                              ① [:, 0, 0, ...]  选收发天线对
            #                                              ② [:l_max]        只取前 l_max 个CIR抽头


            # 计算 RMS 延迟扩展: DS_rms = sqrt( E[τ²] - E[τ]² )
            # 以抽头功率为权重, 值越大表示多径越丰富, 信道频率选择性越强
            h = tf.squeeze(pilots_post_channel, axis=-1)        # [batch, 抽头数]
            delays = tf.range(tf.shape(h)[1], dtype=tf.float32)  # 抽头索引作为等效时延
            power = tf.square(tf.abs(h))                         # 每抽头功率 |h[l]|²
            mean_delay = tf.reduce_sum(delays * power, axis=-1) / tf.reduce_sum(power, axis=-1)
            rms_ds = tf.sqrt(
                tf.reduce_sum(power * tf.square(delays - mean_delay[:, None]), axis=-1)
                / tf.reduce_sum(power, axis=-1)
            )
            rms_ds = tf.expand_dims(rms_ds, -1)                 # [batch, 1]

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
        # 移除第一个 OFDM 符号（被用于导频）
        # Kronecker 导频图案中第 0 个 OFDM 符号的全部子载波都放了导频，不承载数据。
        # 解调后这个符号需要剔除，剩下从索引 1 开始的才是数据符号。
        r_freq = r_freq[:,:,:,1:,:]

        # q 向量广播：对每个子载波做逐元素均衡（类似单抽头均衡）
        # q 来自 _qQ_creator_layer, shape=[batch, fft_size], 是每个子载波的缩放因子
        # 通过 newaxis + tile 广播到 r_freq 同形: [batch, 1, 1, num_data_symbols, fft_size]
        q = q[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        q = tf.tile(q, [1, 1, 1, tf.shape(r_freq)[-2], 1])
        # 逐元素相乘: 每个子载波上的接收符号 × 对应缩放因子 → 单抽头均衡
        # 这是最简单的均衡方式, 假设 Q^H 已大致对角化了信道, 残留只需逐子载波缩放
        r_freq_equalzied = r_freq * q

        # 展平频域符号以匹配 QAM 符号格式
        # r_freq_equalzied: [batch, 1, 1, num_data_symbols, fft_size]
        #   → reshape:       [batch, 1, 1, num_data_symbols * fft_size]
        #   → set_shape:     [None, None, None, tot_symbols_to_deliver]  (固定静态shape供后续层使用)
        current_shape = tf.shape(r_freq_equalzied)
        r_freq_equalzied = tf.reshape(r_freq_equalzied,
                                      [current_shape[0], current_shape[1], current_shape[2], -1])
        r_freq_equalzied.set_shape([None, None, None, self._tot_symbol_to_deliver])

        # 软解调：均衡后的 QAM 符号 → LLR (Log-Likelihood Ratio)
        llr = self._demapper(r_freq_equalzied, no)

        # -------- 步骤 10: 损失计算（仅训练模式） --------
        # BCE-only 训练：只使用二进制交叉熵损失，不使用 PAPR 损失或不确定性权重

        if self.training:
            # BCE 损失：衡量比特传输质量（通信性能）
            bce_loss = tf.squeeze(self.bce(tf.reshape(b, tf.shape(llr)), llr))

            # 总损失 = BCE loss only（简单平均）
            total_loss = tf.reduce_mean(bce_loss)
            bce_mean = tf.reduce_mean(bce_loss)

            # 可视化（可选，仅在 visulaize_progress=True 时启用）
            if self.visulaize_progress:
                self.visulaize(h_freq, Q, rms_ds)

            return total_loss, bce_mean

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
