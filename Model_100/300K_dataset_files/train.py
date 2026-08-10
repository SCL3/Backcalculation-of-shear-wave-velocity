import torch
# print(torch.cuda.is_available())  # Must be True
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # explicit, avoids ambiguity about which backend is active
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import os
import shutil
import csv
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from model import ModelCNN_fvs, ModelResNet34_fvs, ModelResNet50_fvs, ModelDenseNet121_fvs, ModelSwinT_fvs, ModelEfficientNetB0_fvs
from loss_fcns import RMSELoss, RMSE_TV_Loss
from Call_dataset import MyDataset
from torch.utils.data import DataLoader, Subset
from save_validation import save_prediction

# --- Timezone CSV log, run_id uses Taiwan local time ---
TAIWAN_TZ = ZoneInfo("Asia/Taipei")  # UTC+8

# --- Device configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")  # CPU fallback

# --- Program reproducbility Setup ---
def set_seed(seed):
    """Seed torch/cuda/numpy/random, and return a dedicated torch.Generator for reproducible DataLoader shuffling."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    """
    Ensures reproducible randomness inside each DataLoader worker process.
    ! Only needed once random augmentation (Gaussian noise, random truncation) is added in Call_dataset.py.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Logging function ---
def log_epoch_to_csv(log_path, run_id, model_name, criterion, run_started_at, epoch, num_epochs,
                      train_loss, val_loss, is_best, learning_rate,
                      epoch_duration_sec, num_params, batch_size, seed):
    log_parent_dir = os.path.dirname(log_path)
    if log_parent_dir:
        os.makedirs(log_parent_dir, exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id", "model_name", "criterion", "run_started_at", "epoch", "epoch_timestamp",
                "train_loss", "val_loss", "is_best", "learning_rate",
                "epoch_duration_sec", "num_params", "batch_size", "seed"
            ])
        writer.writerow([
            run_id, model_name, criterion, run_started_at, f"{epoch + 1}/{num_epochs}",
            datetime.now(TAIWAN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            f"{train_loss:.6f}", f"{val_loss:.6f}", is_best, f"{learning_rate:.8f}",
            f"{epoch_duration_sec:.2f}", num_params, batch_size, seed
        ])


def train_model(seed, data_folder, in_instances, in_channels, model, model_name,
                add_noise, noise_std, hyperparams, log_path, log_dir):
    """
    Run one full training + validation loop for a single model configuration.
    Call this once per model/config to launch several trainings back to back (e.g. in a loop from __main__).

    Parameters
    ----------
    seed : int
        Random seed for the train/val split and DataLoader shuffling (see note on weight init below).
    data_folder : str
        Path to the dataset folder, expected to contain 'input' and 'output' subfolders.
    in_instances : list[str]
        Names of the input tensors expected by both the dataset and the model (e.g. ['fvs', 'fls', 'x0', 'dx', 'Ch']).
    in_channels : int
        Number of channels of the FVS input image (e.g. 1 or 3).
    model : torch.nn.Module
        Already-instantiated model to train (built by the caller, so its architecture/in_channels are free to vary per run).
    model_name : str
        Free-text label for this run, used in the run_id, the CSV "model_name" column, and the checkpoint filename.
    log_path : str
        Path to the CSV file that accumulates one row per epoch, shared across all runs/models.
    add_noise : bool
        Whether to add Gaussian noise augmentation to the training set only (validation always stays clean/deterministic).
    noise_std : float
        Standard deviation of the Gaussian noise added when add_noise is True.
    hyperparams : list
        [criterion, optimizer, scheduler, train_ratio, batch_size, num_epochs, num_workers, best_error, early_stopping]
        (see the inline comments where this list is unpacked below for what each entry means).
    log_dir : str
        Base directory for TensorBoard logs; this run's events go to log_dir/run_id, so different runs never collide.

    Returns
    -------
    str
        The run_id generated for this run (handy to look it up later in log.csv or pass to analyze_training_log.py).

    Note on reproducibility: the model is already built when it reaches this function, so `seed` here controls the
    data split/shuffling but NOT the model's weight initialization. If you need the weights themselves to be
    reproducible too, call set_seed(seed) yourself right before constructing the model (see the __main__ example).
    """
    # --- Unpack hyperparameters (kept as a list per the caller's request, so comment each slot clearly) ---
    (criterion,  # loss function used for both training and validation
     optimizer,  # optimizer, already bound to model.parameters()
     scheduler,  # learning rate scheduler, already bound to the optimizer above
     train_ratio,  # fraction of the dataset used for training; the rest goes to validation
     batch_size,  # number of samples per DataLoader batch
     num_epochs,  # maximum number of training epochs
     num_workers,  # number of DataLoader worker processes (train and val loaders both use this)
     best_error,  # initial "best validation loss" value, before any epoch has run
     early_stopping  # patience in epochs without val-loss improvement; 0 disables early stopping
     ) = hyperparams

    # --- Reproducibility: reseed RNGs for this run and get a dedicated shuffling generator ---
    g = set_seed(seed)

    # --- Move the model to the target device here, so the caller doesn't have to remember to ---
    model = model.to(device)

    # --- Run identifiers, all timestamped in Taiwan local time ---
    run_started_at = datetime.now(TAIWAN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"{model_name}_{datetime.now(TAIWAN_TZ).strftime('%Y%m%d_%H%M%S')}"
    num_params = sum(p.numel() for p in model.parameters())  # for CSV logging

    # !!! Two separate dataset instances (train & val), so data augmentation can differ between them
    train_dataset_raw = MyDataset(
        in_instances, in_channels,
        os.path.join(data_folder, 'input'), os.path.join(data_folder, 'output100'), # Adapted to Kinh 150K dataset
        add_noise=add_noise, noise_std=noise_std)
    # !!! IMPORTANT : This code will call 90k data. To modify the value, go to Call_dataset.py

    val_dataset_raw = MyDataset(
        in_instances, in_channels,
        os.path.join(data_folder, 'input'), os.path.join(data_folder, 'output100'), # Adapted to Kinh 150K dataset
        add_noise=False)  # !!! validation must always stay clean/deterministic

    # --- Split on indices instead of random_split (shared between both instances) so train/val cover the same files ---
    total_size = len(train_dataset_raw)
    train_size = int(total_size * train_ratio)

    indices = torch.randperm(total_size, generator=g).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_raw, train_indices)
    val_dataset = Subset(val_dataset_raw, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,  # Ensures reproducible random augmentation across DataLoader workers.
        generator=g,  # Ensures reproducible batch shuffling
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- TensorBoard: this run's own subfolder, so parallel/sequential runs never overwrite each other ---
    run_log_dir = os.path.join(log_dir, run_id)
    if os.path.exists(run_log_dir):
        shutil.rmtree(run_log_dir)
    os.makedirs(run_log_dir)
    writer = SummaryWriter(run_log_dir)

    epochs_without_improvement = 0
    print(f"[{run_id}] Early stopping: {'disabled' if early_stopping == 0 else f'patience={early_stopping} epochs'}")

    for epoch in tqdm(range(num_epochs), desc=run_id):
        epoch_start_time = time.time()  # for csv loging

        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, leave=False):
            # Unpack all inputs and targets
            *all_inputs, targets = batch

            # Move everything to device
            all_inputs = [inp.to(device) for inp in all_inputs]
            targets = targets.to(device)
            optimizer.zero_grad()
            # Pass all inputs to the model
            outputs = model(*all_inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False):
                # Unpack all inputs and targets
                *all_inputs, targets = batch
                # Move everything to device
                all_inputs = [inp.to(device) for inp in all_inputs]
                targets = targets.to(device)
                # Pass all inputs to the model
                predicts = model(*all_inputs)
                loss = criterion(predicts, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        is_best = val_loss < best_error
        if is_best:
            best_error = val_loss
            epochs_without_improvement = 0
            os.makedirs("300K_dataset_files/PTH", exist_ok=True)
            torch.save(model.state_dict(), f"300K_dataset_files/PTH/saved_best_model_{run_id}.pth")
            print(f"[{run_id}] Save best model, error", val_loss)
        else:
            epochs_without_improvement += 1

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        print(f"[{run_id}] Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6e}, Time: {epoch_duration:.1f}s")

        # --- Log CSV: one line per epoch, shared file across all runs ---
        log_epoch_to_csv(
            log_path=log_path, run_id=run_id, model_name=model_name, criterion=criterion.__class__.__name__,
            run_started_at=run_started_at, epoch=epoch, num_epochs=num_epochs,
            train_loss=train_loss, val_loss=val_loss, is_best=is_best,
            learning_rate=scheduler.get_last_lr()[0], epoch_duration_sec=epoch_duration,
            num_params=num_params, batch_size=batch_size, seed=seed
        )
        if is_best:
            with torch.no_grad():  # No need to create a graph, so we use torch.no_grad() to save GPU ressources
                idx = 0
                for data in val_dataset:
                    # Unpack all inputs and target
                    *all_inputs, target = data
                    # Add batch dimension and move to device
                    all_inputs = [inp.unsqueeze(0).to(device) for inp in all_inputs]
                    target = target.unsqueeze(0).to(device)
                    # Forward pass with all inputs
                    predict = model(*all_inputs)

                    if idx < 40:
                        input_fvs = all_inputs[0].squeeze(0).cpu().detach() if len(all_inputs) > 0 else None
                        input_x0 = all_inputs[1].flatten().cpu().detach() if len(all_inputs) > 1 else None
                        input_dx = all_inputs[2].flatten().cpu().detach() if len(all_inputs) > 2 else None
                        input_Ch = all_inputs[3].flatten().cpu().detach() if len(all_inputs) > 3 else None
                        folder_save_result = f'300K_dataset_files/validation_results/{run_id}/{"first_epoch" if epoch == 0 else "best_epoch"}'
                        save_prediction(
                            input_fvs=input_fvs,
                            predict=predict.squeeze(0).cpu().detach(),
                            file_id=idx,
                            folder_save_result=folder_save_result,
                            input_x0=[input_x0[0]],
                            input_dx=[input_dx[0]],
                            input_Ch=[input_Ch[0]],
                            target=target.squeeze(0).cpu().detach()
                        )
                    else:
                        break
                    idx += 1

        # --- Early stopping check (runs every epoch, after the CSV row is written) ---
        if early_stopping > 0 and epochs_without_improvement >= early_stopping:
            print(f"[{run_id}] Early stopping: no improvement in val loss for {early_stopping} epochs "
                  f"(best={best_error:.6f}). Stopping at epoch {epoch + 1}/{num_epochs}.")
            break

    # --- Save model ---
    writer.close()
    print(f"[{run_id}] Training complete. Model saved.")
    return run_id


if __name__ == "__main__":
    # --- Check if Cuda exists first ---
    print(torch.__version__)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("No CUDA GPU detected (train.py will fall back to MPS or CPU)")

    # --- Shared settings for this batch of runs ---
    # data_folder = 'training_data_5K/dataset' 5K data
    data_folder = '300K_dataset_files/test/dataset'  # 300K data
    # !!! IMPORTANT : Only 90k data will be used, (change need to be done in Call_dataset.py file)
    # For faster training (1 day min - 2 days max)

    in_instances = ['fvs', 'x0', 'dx', 'Ch']  # New structure on the 300k dataset
    in_channels = 3
    seed = 42

    # Quick sanity check before launching a full training sweep

    dataset = MyDataset(in_instances,
                        in_channels=3,
                        input_dir=os.path.join(data_folder, 'input'),
                        output_dir=os.path.join(data_folder, 'output100'), # Adapted to Kinh 150K dataset
                        add_noise=False,
                        noise_std=0.00)
    sample = dataset[0]
    fvs = sample[0]  # shape (3, H, W): [frequency, phase_velocity, amplitude]

    plt.imshow(fvs[2], aspect='auto')
    plt.colorbar()
    plt.title("Amplitude channel at no noise")
    plt.savefig("noise_check.png", dpi=150, bbox_inches='tight')  # save instead of show
    plt.close()
