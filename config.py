class DeployConfig:
    def __init__(self):
        self.weights = None
        self.data = None
        self.mask = None
        self.k = 0

        self.crop_s = None
        self.crop_e = None

# def deploy_config():
#     config = DeployConfig()
#     config.weights = './out/2023_07_07_10_15_56/e1.pt'
#     config.data = '/cbica/home/lihon/comp_space/bbl_pnc_resting/hcp_sm_data/rs3_sm6_t400_data/rs3_sm6_t400_180432.nii.gz'
#     config.mask = '/cbica/home/lihon/comp_space/bbl_pnc_resting/rnn_autoencoder/scripts/mask_thr0p5_wmparc.2_cc_3mm.nii.gz'
#     config.k = 17
#     config.crop_s = (2, 0, 2)
#     config.crop_e = (58, 72, 58)
#     return config

def deploy_config():
    config = DeployConfig()
    config.weights = './out/2023_08_01_12_24_03/e300.pt'
    config.data = 'data/simtb_data/data/sim_subject_100_DATA.nii'
    config.mask = 'data/simtb_data/mask.nii'
    config.k = 20
    return config

class TrainConfig:
    def __init__(self):
        self.data = None
        self.mask = None
        self.mode = 0               # 0: pretrain, 1: finetune
        self.norm = 0               # 0: global,   1: voxelwise

        self.k = 0
        self.checkpoint = None
        self.epochs = 0
        self.lr = 0.0
        self.tradeoff = 0.0

        self.crop_s = None
        self.crop_e = None

def hcp_pretrain_config():
    config = TrainConfig()
    config.data = '/cbica/home/lihon/comp_space/bbl_pnc_resting/hcp_sm_data/hcp_sm6_t400_tra.txt'
    config.mask = '/cbica/home/lihon/comp_space/bbl_pnc_resting/rnn_autoencoder/scripts/mask_thr0p5_wmparc.2_cc_3mm.nii.gz'
    config.mode = 0
    config.norm = 1
    config.k = 17
    config.epochs = 100
    config.lr = 1e-4
    return config

def hcp_finetune_config():
    config = TrainConfig()
    config.data = '/cbica/home/lihon/comp_space/bbl_pnc_resting/hcp_sm_data/hcp_sm6_t400_tra.txt'
    config.mask = '/cbica/home/lihon/comp_space/bbl_pnc_resting/rnn_autoencoder/scripts/mask_thr0p5_wmparc.2_cc_3mm.nii.gz'
    config.mode = 1
    config.norm = 1
    config.k = 17
    # config.checkpoint = 'weights/hcp_pre_100.pt'
    config.epochs = 100
    config.lr = 1e-4
    config.tradeoff = 10.0
    config.crop_s = (2, 0, 2)
    config.crop_e = (58, 72, 58)
    return config

def simtb_pretrain_config():
    config = TrainConfig()
    config.data = 'data/simtb_data'
    config.mask = 'data/simtb_data/mask.nii'
    config.mode = 0
    config.norm = 1
    config.k = 20
    config.epochs = 300
    config.lr = 1e-4
    return config

def simtb_finetune_config():
    config = TrainConfig()
    config.data = 'data/simtb_data'
    config.mask = 'data/simtb_data/mask.nii'
    config.mode = 1
    config.norm = 1
    config.k = 20
    # config.checkpoint = 'out/2023_07_25_17_53_26/e100.pt'
    config.epochs = 300
    config.lr = 1e-4
    config.tradeoff = 10.0
    return config
