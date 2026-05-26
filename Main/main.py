import sys
sys.path.append('/work_space/project3/Main/src') 
import pickle
import sionna.phy as sn
import tensorflow as tf
import numpy as np
from src.Q_BASELINE.Q_BASELINE_Model import Q_BASELINE_MODEL
from src.Q_Method.Q_Model import Q_MODEL
from src.RQ_Method.RQ_Model import RQ_MODEL
from src.qQ_Method.qQ_Model import qQ_MODEL
from src.OFDM.OFDM_Model import OFDM_MODEL
from src.MC_AE.MC_AE_Model import MC_AE_MODEL
from src.E2EWL.E2EWL_Model import E2EWL_MODEL
from src.SCRRC.SCRRC_Model import SCRRC
from src.SCFDE.SCFDE_Model import SCFDE_MODEL
from config import *
from legends import BER_label_name_dict
from utils.PAPR import emprical_ccdf_plotter, plot_all_ccdf_results, plot_all_ccdf_results_plotly
print(tf.config.list_physical_devices('GPU'))

sn.config.seed = SEED + 4

def prepare_model(model_class, weights_path=None, build_args=(1, 0.0), **kwargs):
    r'''
        Load weights of a model class, if exsits
    '''
    model = model_class(**kwargs)
    model(*build_args)  # call the model once to build so weights could be loaded
    if weights_path:
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
            model.set_weights(weights)
    return model


def main():

    ###############################
    # Benchmarking
    ###############################
    Models = {
    # 'MC-AE MODEL Perfect CSI':          prepare_model(MC_AE_MODEL,'weights-MC_AE', true_channel_as_decoder_input=True),
    # 'MC-AE MODEL Imperfect CSI':        prepare_model(MC_AE_MODEL,'weights-MC_AE'),
    'SC/FDE Model':                     prepare_model(SCFDE_MODEL),    
    'OFDM Model':                       prepare_model(OFDM_MODEL),
    # 'SCRRC Model':                      prepare_model(SCRRC),
    # 'E2EWL AWGN Model':                 prepare_model(E2EWL_MODEL,'weights-E2EWL_AWGN', is_multypath=False),
    'E2EWL MP Model':                   prepare_model(E2EWL_MODEL,'weights-E2EWL_MP', is_multypath=True),
    # 'Q BaseLine Model':                 prepare_model(Q_BASELINE_MODEL),
    # 'Q Method Model':                   prepare_model(Q_MODEL,'weights-Q_Method'),    
    # 'RQ Method Model':                  prepare_model(RQ_MODEL,'weights-RQ_Method'),      
    # 'qQ Method Model':                  prepare_model(qQ_MODEL,'weights-qQ_Method'),   
    'qQ Method Model':                  prepare_model(qQ_MODEL,'weights-qQ_Method_Final'),      
    }

    ###############################
    # Evaluating
    ###############################   
    ber_plots = sn.utils.PlotBER("")
    for model_name, model in Models.items():
        # ber_plots.simulate(model,
        #             ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 40),
        #             batch_size=BATCH_SIZE*100,
        #             num_target_block_errors=100000, # simulate until 100 block errors occured
        #             legend=BER_label_name_dict[model_name],
        #             soft_estimates=True,
        #             max_mc_iter=10000, # run 100 Monte-Carlo simulations (each with batch_size samples)
        #             show_fig=False)
        ber_plots.simulate(model,
                        ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 25),
                        batch_size=BATCH_SIZE*200,
                        num_target_block_errors=1000,  # reduced for speed; increase for smoother tail
                        legend=BER_label_name_dict[model_name],
                        soft_estimates=False,
                        max_mc_iter=2000,
                        show_fig=False)

    # Saving Fig
    ber_plots(save_fig=True,path='BerPlot.png',ylim=(1e-3/3,1))

    ###############################
    # CCDF
    ############################### 
    # exclude = {"E2EWL MP Model", "SC/FDE Model"}
    # filtered_Models = {k: v for k, v in Models.items() if k not in exclude}
    # ccdf_results = {}
    # for model_name, model in filtered_Models.items():    
    #     model.CCDF_mode = True
    #     x_time, rms_ds = model(205,0)
    #     ccdf_results[model_name] = emprical_ccdf_plotter(x_time, rms_ds)
    
    # plot_all_ccdf_results(ccdf_results)

    # plot_all_ccdf_results_plotly(ccdf_results) # HTML CCDF plot viewer

if __name__ == "__main__":
    main()
    