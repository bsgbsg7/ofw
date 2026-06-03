# BER plots legends
BER_label_name_dict = { 'OFDM Model': 'OFDM',
                        'qQ Method Model': 'DeepOFW',
                        'SC/FDE Model': 'SC/FDE',
                        'E2EWL MP Model': 'E2EWL (DL inference required)'}

# I modified sionna source code to plot the ber curve with markers and line width, and changed '(dB)' to '[dB]' in the axis labels. 
# to reconstrcut the BER plot incluidng markers and line width, change the sionna sourcecode plotting script:
# go to sionna.phy.utils.plotting.py in the function plot_ber where the plt.semilogy (or F12 the ber_plots call method, to see it)

# change from this (line 115):
# plt.semilogy(snr_db[idx], b, line_style, linewidth=2)

# to this
# plt.semilogy(snr_db[idx], b, line_style, marker=['o', 's', '^', 'd'][idx], markevery=4, markersize=12, linewidth=2.5)

# also change '(dB)' to '[dB]'


# CCDF plots legends
CCDF_label_name_dict = { 'OFDM Model': 'OFDM',
                        'qQ Method Model': None,
                        'SC/FDE Model': 'SC/FDE',
                        'E2EWL MP Model': 'E2EWL (DL inference required)'}

CCDF_marker_dict     = {'OFDM Model': '*',
                        'qQ Method Model': None,
                        'SC/FDE Model': '<',
                        'E2EWL MP Model':'+'}
