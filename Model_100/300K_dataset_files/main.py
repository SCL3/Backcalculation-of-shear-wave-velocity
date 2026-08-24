import argparse
import torch
import torch.optim as optim

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
from loss_fcns import RMSELoss

# =====================================================================
# GLOBAL CONFIGURATION !!! Change the config if needed train/test mode; seed; instances ...
# =====================================================================

MODE = "train"  # "train" or "test" (Overriden with : python main.py --mode train)

SEED = 42

# Input structure used to TRAIN the models stored in PTH/ (300K dataset structure).
# !!! Must match exactly what was used at training time, otherwise
# model.load_state_dict() will fail with a size-mismatch error.
IN_INSTANCES = ['fvs', 'x0', 'dx', 'Ch']
IN_CHANNELS = 3

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
    # !!! IMPORTANT : Only 90k data will be used (change to be done in Call_dataset.py)
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
    """
    set_seed(SEED)
    V2_all_geo_film = ModelCNN_fvs_v2_all_geo_film(IN_INSTANCES, IN_CHANNELS, 16, 128, fusion_seed=SEED)

    """
    set_seed(SEED)  # TO RESTART
    Resnet50_v2_conf1 = ModelResNet50_fvs_v2(IN_INSTANCES, IN_CHANNELS, 32, 128)
    set_seed(SEED)
    V2_all_geo = ModelCNN_fvs_v2_all_geo(IN_INSTANCES, IN_CHANNELS, 16, 128, fusion_seed=SEED)
    set_seed(SEED)
    V2_film = ModelCNN_fvs_v2_film(IN_INSTANCES, IN_CHANNELS, 16, 128, fusion_seed=SEED)
    """

    # Noise tests: std = 0.0 OK / 0.01 NO / 0.02 OK / 0.05 OK / 0.08 OK / 0.1 OK
    models = [
        # (ModelCNN, "ModelCNN_fvs_std0.08_90k_RMSELoss"),
        # (Resnet50, "Resnet50_fvs_std0.08_90k_RMSELoss"),
        # (Densenet121, "Densenet121_std0.08_90k_RMSELoss"),
        # (ModelSwinT, "ModelSwinT_std0.08_90k_RMSELoss"),
        # (ModelEfficientNetB0, "ModelEfficientNetB0_fvs_std0.08_90k_RMSELoss"),
        # (ModelCustomCNN, "ModelCustomCNN_fvs_No_Noise_90k_RMSELoss"),
        # (ModelCNN_v2_conf1, "ModelCNN_fvs_v2_32_128_No_Noise_90k_RMSELoss"),
        # (ModelCNN_v2_conf2, "ModelCNN_fvs_v2_16_16_No_Noise_90k_RMSELoss"),
        # (ModelCNN_v2_conf3, "ModelCNN_fvs_v2_32_16_No_Noise_90k_RMSELoss"),
        # (ModelCNN_v2_conf4, "ModelCNN_fvs_v2_16_128_No_Noise_90k_RMSELoss"),

        (V2_all_geo_film, "ModelCNN_fvs_v2_all_geo_film_No_Noise_90k_RMSELoss"),
        # (V2_all_geo, "ModelCNN_fvs_v2_all_geo_No_Noise_90k_RMSELoss"),
        # (V2_film, "ModelCNN_fvs_v2_film_No_Noise_90k_RMSELoss"),
        # (Resnet50_v2_conf1, "Resnet50_fvs_v2_32_128_90k_RMSELoss"),
    ]

    for model, model_name in models:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # !!! old value lr = 0.0001
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        hyperparams = [
            RMSELoss(),  # loss function
            optimizer,  # optimizer bound to this model's parameters
            scheduler,  # LR scheduler bound to the optimizer above
            0.8,  # train_ratio
            8,  # batch_size  !!! old value : 8
            200,  # num_epochs  !!! old value : 200
            4,  # num_workers  !!! old value : 4
            999999,  # best_error (initial value)
            45,  # early_stopping (0 = disabled)  !!! old value : 45
        ]

        print(f"BEGIN TRAINING OF [{model_name}] WITHOUT noise V2 -----------------")
        train_model(
            seed=SEED,
            data_folder=data_folder,
            in_instances=IN_INSTANCES,
            in_channels=IN_CHANNELS,
            model=model,
            model_name=model_name,
            log_path="300K_dataset_files/log/90K_RMSELoss_No_Noise/log_all_models_v2.csv",
            add_noise=False,
            noise_std=0.01,
            hyperparams=hyperparams,
            log_dir="300K_dataset_files/runs",
        )
        print(f"END TRAINING OF [{model_name}] WITHOUT noise V2 -----------------")


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