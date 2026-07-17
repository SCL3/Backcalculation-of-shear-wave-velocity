import torch
import torch.optim as optim

from train import set_seed, train_model
from model import ModelCNN_fvs, ModelResNet34_fvs, ModelResNet50_fvs, ModelDenseNet121_fvs, ModelSwinT_fvs, ModelEfficientNetB0_fvs
from loss_fcns import RMSELoss

if __name__ == "__main__":
    # --- Check if Cuda exists first ---
    print(torch.__version__);
    print(torch.cuda.is_available());
    print(torch.cuda.get_device_name(0))

    # --- Shared settings for this batch of runs ---
    data_folder = "5K Dataset Files/training_data_5K/dataset"  # 150K data

    in_instances = ['fvs', 'fls', 'x0', 'dx', 'Ch']
    in_channels = 3
    seed = 42

    # --- Models and name ---
    # !!!!! RESEED right before each construction so every model's newly-added layers
    set_seed(seed)
    ModelCNN = ModelCNN_fvs(in_instances, in_channels)
    """
    set_seed(seed)
    Resnet50 = ModelResNet50_fvs(in_instances, in_channels)
    set_seed(seed)
    Resnet34 = ModelResNet34_fvs(in_instances, in_channels)
    
    set_seed(seed)
    Densenet121 = ModelDenseNet121_fvs(in_instances, in_channels)
    set_seed(seed)
    ModelSwinT = ModelSwinT_fvs(in_instances, in_channels)
    
    set_seed(seed)
    ModelEfficientNetB0 = ModelEfficientNetB0_fvs(in_instances, in_channels)
    """

    models = [
        (ModelCNN, "BASELINE_ModelCNN_fvs"),
        #(Resnet50, "Resnet50_fvs"),
        #(Resnet34, "Resnet34_fvs"),
        #(Densenet121, "Densenet121_fvs"),
        #(ModelSwinT, "ModelSwinT_fvs"),
        # (ModelEfficientNetB0, "ModelEfficientNetB0_fvs_Noise_std0.05"),
    ]

    # --- Run the training each model one after another ---
    for model, model_name in models:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # --- Hyperparameters ---
        hyperparams = [
            RMSELoss(),  # loss function
            optimizer,  # optimizer bound to this model's parameters
            scheduler,  # LR scheduler bound to the optimizer above
            0.8,  # train_ratio
            8,  # batch_size
            200,  # num_epochs
            4,  # num_workers
            999999,  # best_error (initial value)
            45,  # early_stopping (0 = disabled)
        ]

        print(f"BEGIN TRAINING OF [{model_name}] -----------------")
        train_model(
            seed=seed,
            data_folder=data_folder,
            in_instances=in_instances,
            in_channels=in_channels,
            model=model,
            model_name=model_name,
            log_path="5K Dataset Files/log/RMSELoss_No_Noise/log_all_models.csv",
            add_noise=False,
            noise_std=0.05,  # Gotta test on 0.02, 0.05 and 0.1
            hyperparams=hyperparams,
            log_dir="5K Dataset Files/runs",
        )
        print(f"END TRAINING OF [{model_name}] -----------------")
