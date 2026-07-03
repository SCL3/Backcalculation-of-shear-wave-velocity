
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device used :", device)

from model import ModelCNN_fvs
from loss_fcns import RMSELoss
from Call_dataset import MyDataset
from torch.utils.data import DataLoader, random_split
from save_validation import save_prediction
import os
import shutil

data_folder = 'training_data_5K/dataset'
in_instances = ['fvs', 'fls', 'x0', 'dx', 'Ch']
in_channels = 3
model = ModelCNN_fvs(in_instances, in_channels).to(device)
if __name__ == "__main__":
    dataset = MyDataset(in_instances,
                        in_channels,
                        os.path.join(data_folder, 'input'),
                        os.path.join(data_folder, 'output')
                        )
    # Model hyperparameters
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_ratio = 0.8
    batch_size = 8
    num_epochs = 3
    best_error = 999999

    # Data splitting
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    log_dir = "runs/csv_regression"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(num_epochs)):
        # Training
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

        # Validation
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

        if val_loss < best_error:
            best_error = val_loss
            torch.save(model.state_dict(), "saved_best_model.pth")
            print("Save best model, error", val_loss)
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6e}")
        idx = 0

        for data in val_dataset:
            # Unpack all inputs and target
            *all_inputs, target = data
            # Add batch dimension and move to device
            all_inputs = [inp.unsqueeze(0).to(device) for inp in all_inputs]
            target = target.unsqueeze(0).to(device)
            # Forward pass with all inputs
            predict = model(*all_inputs)

            if idx < 50:
                input_fvs = all_inputs[0].squeeze(0).cpu().detach() if len(all_inputs) > 0 else None
                input_fls = all_inputs[1].squeeze(0).cpu().detach() if len(all_inputs) > 1 else None
                input_x0 = all_inputs[2].flatten().cpu().detach() if len(all_inputs) > 2 else None
                input_dx = all_inputs[3].flatten().cpu().detach() if len(all_inputs) > 3 else None
                input_Ch = all_inputs[4].flatten().cpu().detach() if len(all_inputs) > 4 else None
                save_prediction(
                    input_fvs=input_fvs,
                    input_fls=input_fls,
                    predict=predict.squeeze(0).cpu().detach(),
                    file_id=idx,
                    folder_save_result='validation_results',
                    input_x0=[input_x0[0]],
                    input_dx=[input_dx[0]],
                    input_Ch=[input_Ch[0]],
                    target=target.squeeze(0).cpu().detach()
                )
            else:
                break
            idx += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    writer.close()
    print("Training complete. Model saved.")