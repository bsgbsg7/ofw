# DeepOFW

##
This repository contains the implementation, training, and evaluation framework for multiple algorithms organized under a unified project structure of the DeepOFW paper. The `Main/` directory includes the primary execution script for evaluation as well as the subdirectories of all supported algorithms. Running `main.py` from `Main/` generates performance outputs such as the BER curve and CCDF. Each algorithm can be trained independently through its own `train.py` script located in the corresponding subdirectory. To simplify installation and ensure reproducibility, `requirements.txt` and `Dockerfile` are provided.

https://arxiv.org/abs/2603.23544

## Abstract
Peak-to-average power ratio (PAPR) remains a major limitation of multicarrier modulation schemes such as orthogonal frequency-division multiplexing (OFDM), reducing power amplifier efficiency and limiting practical transmit power. In this work, we propose \emph{DeepOFW}, a deep learning–driven OFDM-flexible waveform modulation framework that enables data-driven waveform design while preserving the low-complexity hardware structure of conventional transceivers. The proposed architecture is fully differentiable, allowing end-to-end optimization of waveform generation and receiver processing under practical physical constraints. Unlike neural transceiver approaches that require deep learning inference at both ends of the link, DeepOFW confines the learning stage to an offline or centralized unit, enabling deployment on standard transmitter and receiver hardware without additional computational overhead. The framework jointly optimizes waveform representations and detection parameters while explicitly incorporating PAPR constraints during training. Extensive simulations over 3GPP multipath channels demonstrate that the learned waveforms significantly reduce PAPR compared with classical OFDM while simultaneously improving bit error rate (BER) performance relative to state-of-the-art transmission schemes. These results highlight the potential of data-driven waveform design to enhance multicarrier communication systems while maintaining hardware-efficient implementations. An open-source implementation of the proposed framework is released to facilitate reproducible research and practical adoption.


## Repository Structure

```text
.
├── Main/
│   ├── main.py
│   ├── E2EWL/
│   │   └── train.py
│   │   └── ...
│   ├── DeepOFW/
│   │   └── train.py
│   │   └── ...
│   └── ...
├── Dockerfile
└── requirements.txt
