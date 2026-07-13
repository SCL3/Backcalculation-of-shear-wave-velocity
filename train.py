import torch
# print(torch.cuda.is_available())  # Must be True
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import os
import shutil
import csv
import time
from datetime import datetime

from model import ModelResNet34_fvs, ModelResNet50_fvs, ModelDenseNet121_fvs
from loss_fcns import RMSELoss, RMSE_TV_Loss
from Call_dataset import MyDataset
from torch.utils.data import DataLoader, Subset
from save_validation import save_prediction

# --- Program reproducbility Setup ---
g = torch.Generator()
def set_seed(seed=42):
    """
    Define the seed of the program to ensure reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g.manual_seed(seed)

SEED = 42
set_seed(SEED)

def seed_worker(worker_id):
    """
    Ensures reproducible randomness inside each DataLoader worker process.
    ! Only needed once random augmentation (Gaussian noise, random truncation) is added in Call_dataset.py.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Logging function ---
def log_epoch_to_csv(log_path, run_id, model_name, criterion, run_started_at, epoch, num_epochs,
                      train_loss, val_loss, is_best, learning_rate,
                      epoch_duration_sec, num_params, batch_size, seed):
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
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{train_loss:.6f}", f"{val_loss:.6f}", is_best, f"{learning_rate:.8f}",
            f"{epoch_duration_sec:.2f}", num_params, batch_size, seed
        ])

# --- Device configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")  # CPU fallback

data_folder = 'training_data_5K/dataset'
in_instances = ['fvs', 'fls', 'x0', 'dx', 'Ch']
in_channels = 3
model = ModelResNet34_fvs(in_instances, in_channels).to(device)  # ModelResNet50_fvs, ModelResNet34_fvs
MODEL_NAME = model.__class__.__name__   # For csv loging
NUM_PARAMS = sum(p.numel() for p in model.parameters())  # For csv loging

if __name__ == "__main__":
    # --- CSV loging ---
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = "log.csv"   # One file for all the model

    # !!! Two separate dataset instances (train & val), so data augmentation (for later) can differ between them
    train_dataset_raw = MyDataset(
        in_instances,
        in_channels,
        os.path.join(data_folder, 'input'),
        os.path.join(data_folder, 'output'),
        add_noise=False,  # Try to see if adding noise can make the validation results better
        noise_std=0.02)

    val_dataset_raw = MyDataset(
        in_instances,
        in_channels,
        os.path.join(data_folder, 'input'),
        os.path.join(data_folder, 'output'),
        add_noise=False)    # !!! validation must always stay clean/deterministic

    # --- Model hyperparameters --- (Use Optuna later on the best Model)
    criterion = RMSELoss() #  RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_ratio = 0.8
    batch_size = 8
    num_epochs = 200 # 150
    best_error = 999999

    # --- Split on indices instead of random_split (shared between both instances) so train/val cover the same files ---
    total_size = len(train_dataset_raw)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    indices = torch.randperm(total_size, generator=g).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_raw, train_indices)
    val_dataset = Subset(val_dataset_raw, val_indices)

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # old code
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,  # Ensures reproducible random augmentation (Gaussian noise, random truncation) across DataLoader workers.
        generator=g,  # Ensures reproducible for batches shuffling
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    log_dir = f"runs/{run_id}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()  # for csv loging

        # --- Training ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
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
            for batch in tqdm(val_loader):
                # Unpack all inputs and targets
                *all_inputs, targets = batch
                # Move everything to device
                all_inputs = [inp.to(device) for inp in all_inputs]
                targets = targets.to(device)
                # Pass all inputs to the model
                predicts = model(*all_inputs)
                loss = criterion(predicts, targets)
                val_loss += loss.item()
        print("")
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        is_best = val_loss < best_error
        if is_best:
            best_error = val_loss
            torch.save(model.state_dict(), f"saved_best_model_{run_id}.pth")
            print("Save best model, error", val_loss)

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6e}, Time: {epoch_duration:.1f}s")

        # --- Log CSV : One line per epoch ---
        log_epoch_to_csv(
            log_path=log_path, run_id=run_id, model_name=MODEL_NAME, criterion=criterion.__class__.__name__,
            run_started_at=run_started_at, epoch=epoch, num_epochs=num_epochs,
            train_loss=train_loss, val_loss=val_loss, is_best=is_best,
            learning_rate=scheduler.get_last_lr()[0], epoch_duration_sec=epoch_duration,
            num_params=NUM_PARAMS, batch_size=batch_size, seed=SEED
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

                    if idx < 30:
                        input_fvs = all_inputs[0].squeeze(0).cpu().detach() if len(all_inputs) > 0 else None
                        input_fls = all_inputs[1].squeeze(0).cpu().detach() if len(all_inputs) > 1 else None
                        input_x0 = all_inputs[2].flatten().cpu().detach() if len(all_inputs) > 2 else None
                        input_dx = all_inputs[3].flatten().cpu().detach() if len(all_inputs) > 3 else None
                        input_Ch = all_inputs[4].flatten().cpu().detach() if len(all_inputs) > 4 else None
                        if epoch == 0:
                            save_prediction(
                                input_fvs=input_fvs,
                                input_fls=input_fls,
                                predict=predict.squeeze(0).cpu().detach(),
                                file_id=idx,
                                folder_save_result=f'validation_results/{run_id}/first_epoch',
                                input_x0=[input_x0[0]],
                                input_dx=[input_dx[0]],
                                input_Ch=[input_Ch[0]],
                                target=target.squeeze(0).cpu().detach()
                            )
                        else:
                            save_prediction(
                                input_fvs=input_fvs,
                                input_fls=input_fls,
                                predict=predict.squeeze(0).cpu().detach(),
                                file_id=idx,
                                folder_save_result=f'validation_results/{run_id}/best_epoch',
                                input_x0=[input_x0[0]],
                                input_dx=[input_dx[0]],
                                input_Ch=[input_Ch[0]],
                                target=target.squeeze(0).cpu().detach()
                            )
                    else:
                        break
                    idx += 1

    # --- Save model ---
    writer.close()
    print("Training complete. Model saved.")