import argparse
import torch
import torch.optim as optim
import math

from train import set_seed, train_model
from test import run_test
from model import (
    ModelCNN_fvs,
    ModelCNN_fvs_v2,
    ModelResNet34_fvs,
    ModelResNet50_fvs,
    ModelResNet50_fvs_v2,
    ModelDenseNet121_fvs,
    ModelSwinT_fvs,
    ModelEfficientNetB0_fvs,
    ModelCustomCNN_fvs,
    ModelCNN_fvs_v2_all_geo_film,
    ModelCNN_fvs_v2_film,
    ModelCNN_fvs_v2_all_geo
)
from model_test_geo import ModelCNN_fvs_v3_geo, ModelDenseNet121_fvs_geo, ModelEfficientNetB0_fvs_geo

from loss_fcns import RMSELoss

# =====================================================================
# GLOBAL CONFIGURATION !!! Change the config if needed train/test mode; seed; instances ...
# =====================================================================

MODE = "train"  # "train" or "test" (Overriden with : python main.py --mode train)

SEED = 42  # !!! old value : 42

# Input structure used to TRAIN the models stored in PTH/ (300K dataset structure).
# !!! Must match exactly what was used at training time, otherwise
# model.load_state_dict() will fail with a size-mismatch error.
IN_INSTANCES = ['fvs', 'x0', 'dx', 'Ch']
IN_CHANNELS = 3

TRAIN_RATIO = 0.8
NUM_EPOCHS = 250
EARLY_STOPPING  = 25
BATCH_SIZE      = 64
NUM_WORKERS = 4

LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4   # AdamW defaults to 1e-2 -> always set this explicitly

COSINE_HORIZON  = 120    # horizon the LR schedule is actually designed for
WARMUP_EPOCHS   = 5
ETA_MIN         = 1e-6   # LR floor; the schedule stays flat here past COSINE_HORIZON
MIN_DELTA_REL   = 1e-3   # an improvement must beat the best by >0.1% to reset patience

# =====================================================================
# TEST CONFIGURATION
# =====================================================================
PTH_ROOT = "300K_dataset_files/PTH"                       # root folder containing all the .pth subfolders
TEST_DATA_FOLDER = "300K_dataset_files/testing_measured"  # contains "input" with the 3 real-field .mat files
TEST_CASE = "measured"                 # "measured" (no target) or "synthetic" (target available)

# =====================================================================
# TRAIN MODE
# =====================================================================
def run_train():
    # data_folder = 'training_data_5K/dataset'  # 5K data
    data_folder = r'C:\Users\KINH\training_data\dataset2'  # 300K data
    # !!! IMPORTANT : 300k data will be used (change to be done in Call_dataset.py)
    # For faster training (1 day min - 2 days max)

    # !!!!! RESEED right before each construction so every model gets identical init
    """
    set_seed(SEED)
    ModelCNN = ModelCNN_fvs(IN_INSTANCES, IN_CHANNELS)
    set_seed(SEED)
    Resnet50 = ModelResNet50_fvs(IN_INSTANCES, IN_CHANNELS)
    set_seed(SEED)
    Densenet121 = ModelDenseNet121_fvs(IN_INSTANCES, IN_CHANNELS)
    set_seed(SEED)
    ModelSwinT = ModelSwinT_fvs(IN_INSTANCES, IN_CHANNELS)
    set_seed(SEED)
    ModelEfficientNetB0 = ModelEfficientNetB0_fvs(IN_INSTANCES, IN_CHANNELS)
    set_seed(SEED)
    ModelCustomCNN = ModelCustomCNN_fvs(IN_INSTANCES, IN_CHANNELS)
    set_seed(SEED)
    ModelCNN_v2_conf1 = ModelCNN_fvs_v2(IN_INSTANCES, IN_CHANNELS, 32, 128)
    set_seed(SEED)
    ModelCNN_v2_conf2 = ModelCNN_fvs_v2(IN_INSTANCES, IN_CHANNELS, 16, 16)
    set_seed(SEED)
    ModelCNN_v2_conf3 = ModelCNN_fvs_v2(IN_INSTANCES, IN_CHANNELS, 32, 16)
    set_seed(SEED)
    ModelCNN_v2_conf4 = ModelCNN_fvs_v2(IN_INSTANCES, IN_CHANNELS, 16, 128)
    
    set_seed(SEED)
    V2_all_geo_film = ModelCNN_fvs_v2_all_geo_film(IN_INSTANCES, IN_CHANNELS, 16, 128, fusion_seed=SEED)

    Resnet50_v2_conf1 = ModelResNet50_fvs_v2(IN_INSTANCES, IN_CHANNELS, 32, 128)
    set_seed(SEED)
    V2_all_geo = ModelCNN_fvs_v2_all_geo(IN_INSTANCES, IN_CHANNELS, 16, 128, fusion_seed=SEED)
    set_seed(SEED)
    V2_film = ModelCNN_fvs_v2_film(IN_INSTANCES, IN_CHANNELS, 16, 128, fusion_seed=SEED)
    
    set_seed(SEED)
    V3_geo_G1 = ModelCNN_fvs_v3_geo(IN_INSTANCES, IN_CHANNELS, 64, 128,
                                    num_fourier_bands=0, use_film=False, fusion_seed=SEED)
    set_seed(SEED)
    V3_geo_G2 = ModelCNN_fvs_v3_geo(IN_INSTANCES, IN_CHANNELS, 64, 128,
                                    num_fourier_bands=4, use_film=False, fusion_seed=SEED)
    set_seed(SEED)
    V3_geo_G3 = ModelCNN_fvs_v3_geo(IN_INSTANCES, IN_CHANNELS, 64, 128,
                                    num_fourier_bands=4, use_film=True, fusion_seed=SEED)
    """

    set_seed(SEED)
    Dense_geo = ModelDenseNet121_fvs_geo(IN_INSTANCES, IN_CHANNELS, 64, 128, fusion_seed=SEED)
    set_seed(SEED)
    Eff_geo = ModelEfficientNetB0_fvs_geo(IN_INSTANCES, IN_CHANNELS, 64, 128, fusion_seed=SEED)

    # Noise tests: std = 0.0 OK / 0.01 NO / 0.02 OK / 0.05 OK / 0.08 OK / 0.1 OK
    models = [
        # (ModelCNN, "ModelCNN_fvs_std0.08_90k_RMSELoss"),
        # (Resnet50, "Resnet50_fvs_std0.08_90k_RMSELoss"),
        # (Densenet121, "Densenet121_std0.08_90k_RMSELoss"),
        # (ModelSwinT, "ModelSwinT_std0.08_90k_RMSELoss"),
        # (ModelEfficientNetB0, "ModelEfficientNetB0_fvs_std0.08_90k_RMSELoss"),
        # (ModelCustomCNN, "ModelCustomCNN_fvs_No_Noise_5k_RMSELoss"),
        # (ModelCNN_v2_conf1, "ModelCNN_fvs_v2_32_128_No_Noise_5k_RMSELoss"),
        # (ModelCNN_v2_conf2, "ModelCNN_fvs_v2_16_16_No_Noise_5k_RMSELoss"),
        # (ModelCNN_v2_conf3, "ModelCNN_fvs_v2_32_16_No_Noise_5k_RMSELoss"),
        # (ModelCNN_v2_conf4, "ModelCNN_fvs_v2_16_128_No_Noise_5k_RMSELoss"),
        # (V2_all_geo_film, "ModelCNN_fvs_v2_all_geo_film_No_Noise_5k_RMSELoss"),
        # (V2_all_geo, "ModelCNN_fvs_v2_all_geo_No_Noise_5k_RMSELoss"),
        # (V2_film, "ModelCNN_fvs_v2_film_No_Noise_5k_RMSELoss"),
        # (Resnet50_v2_conf1, "Resnet50_fvs_v2_32_128_5k_RMSELoss"),

        #(V3_geo_G1, "ModelCNN_fvs_v3_geo_G1_90k_RMSELoss"),
        # (V3_geo_G2, "ModelCNN_fvs_v3_geo_G2_90k_RMSELoss"),
        #(V3_geo_G3, "ModelCNN_fvs_v3_geo_G3_90k_RMSELoss"),

        (Dense_geo, "ModelDenseNet121_fvs_geo_300k_RMSELoss"),
        (Eff_geo, "ModelEfficientNetB0_fvs_geo_300k_RMSELoss"),
    ]

    for model, model_name in models:
        # AdamW decouples weight decay from the adaptive step, unlike Adam which
        # folds it into the gradient and divides it by sqrt(v_hat).
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Linear warmup, then cosine decay, CLAMPED flat at ETA_MIN past COSINE_HORIZON.
        # CosineAnnealingLR on its own is periodic (period 2*T_max)
        def _make_lr_lambda(warmup, horizon, floor_ratio):
            def f(epoch):
                if epoch < warmup:
                    return 0.1 + 0.9 * epoch / warmup  # linear warmup: 0.1 -> 1.0
                t = min(epoch - warmup, horizon - warmup) / (horizon - warmup)
                return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

            return f

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(WARMUP_EPOCHS, COSINE_HORIZON, ETA_MIN / LEARNING_RATE))

        hyperparams = [
            RMSELoss(),  # loss function
            optimizer,  # optimizer bound to this model's parameters
            scheduler,  # LR scheduler bound to the optimizer above
            TRAIN_RATIO,  # train_ratio
            BATCH_SIZE,  # batch_size  !!! old value : 8
            NUM_EPOCHS,  # num_epochs  !!! old value : 200
            NUM_WORKERS,  # num_workers  !!! old value : 4
            999999,  # best_error (initial value)
            EARLY_STOPPING,  # early_stopping (0 = disabled)  !!! old value : 45
        ]

        print(f"BEGIN TRAINING OF [{model_name}] 90K WITHOUT noise V3 -----------------")
        train_model(
            seed=SEED,
            data_folder=data_folder,
            in_instances=IN_INSTANCES,
            in_channels=IN_CHANNELS,
            model=model,
            model_name=model_name,
            log_path="300K_dataset_files/log/300K_RMSELoss_No_Noise/log_300k_v1.csv",
            add_noise=False,
            noise_std=0.01,
            hyperparams=hyperparams,
            log_dir="300K_dataset_files/runs",
            min_delta_rel=MIN_DELTA_REL,
            resume=True,
        )
        print(f"END TRAINING OF [{model_name}] 90K WITHOUT noise V3 -----------------")


# =====================================================================
# MAIN FUNCTION
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models, or test every .pth in PTH/")
    parser.add_argument("--mode", choices=["train", "test"], default=MODE)
    args = parser.parse_args()

    # --- Environment check ---
    print(torch.__version__)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        print("No CUDA GPU detected (train.py will fall back to MPS)")
    else:
        print("No CUDA GPU detected (train.py will fall back to CPU)")

    if args.mode == "train":
        run_train()
    else:
        run_test(
            pth_root=PTH_ROOT,
            data_folder=TEST_DATA_FOLDER,
            in_instances=IN_INSTANCES,
            in_channels=IN_CHANNELS,
            case=TEST_CASE,
        )