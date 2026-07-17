import scipy.io
import torch
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, in_instances, in_channels, input_dir, output_dir=None, add_noise=False, noise_std=0.02):
        self.input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mat')]
        self.input = input_dir
        self.in_instances = in_instances
        self.in_channels = in_channels
        self.add_noise = add_noise
        self.noise_std = noise_std  # std of Gaussian noise, relative to the normalized amplitude (max=1)

        # Only set up output files if not in test mode
        if output_dir is not None:
            # self.output_files = [file.replace(main_input_dir, output_dir) for file in self.main_input_files]
            self.output_dir = output_dir
            self.output_files=[]
            for main_file in self.input_files:
                # Extract the file index (assuming files are named like "0.mat", "1.mat", etc.)
                file_index=os.path.splitext(os.path.basename(main_file))[0]
                output_file=os.path.join(output_dir,f"{file_index}.csv")
                self.output_files.append(output_file)
        else:
            self.output_dir = None
            self.output_files = None

    def _apply_gaussian_noise(self, amplitude):
        """Simulate measurement noise on the amplitude spectrum. Left untouched (and
        deterministic) when add_noise=False, so it is safe to call unconditionally."""
        if not self.add_noise or self.noise_std <= 0:
            return amplitude
        noise = np.random.normal(0.0, self.noise_std, size=amplitude.shape).astype(amplitude.dtype)
        return np.clip(amplitude + noise, 0.0, None)

    def __len__(self):
        return min(90000, len(self.input_files))
    def __getitem__(self, idx):
        mat_data = scipy.io.loadmat(self.input_files[idx])
        var_names = [key for key in mat_data.keys() if not key.startswith('__')]
        if not var_names:
            raise ValueError(f"No data variables found in {self.input_files[idx]}")
        struct_data = mat_data[var_names[0]]
        var_fields = struct_data.dtype.names
        input_tensors = []
        fvs = struct_data[var_fields[0]][0, 0]  # [0,0] because MATLAB structs become arrays

        if self.in_channels==1:
            amplitude = fvs[-1]
            amplitude = self._apply_gaussian_noise(amplitude)
            fvs[-1] = amplitude# / amplitude.max(axis=1, keepdims=True)
            main_input_tensor_fvs = torch.tensor(fvs[-1], dtype=torch.float32).unsqueeze(0)
        else:
            frequency, phase_velocity, amplitude = fvs[0], fvs[1], fvs[-1]
            fvs[0] = frequency# / frequency.max(axis=0, keepdims=True)
            fvs[1] = phase_velocity# / phase_velocity.max(axis=1, keepdims=True)
            amplitude = amplitude / amplitude.max(axis=1, keepdims=True)
            fvs[-1] = self._apply_gaussian_noise(amplitude)
            main_input_tensor_fvs = torch.tensor(fvs, dtype=torch.float32)
            # main_input_tensor = torch.tensor([fvs[0], fvs[-1]], dtype=torch.float32)
        input_tensors.append(main_input_tensor_fvs)

        """
        # !!! No fls in the 150K Dataset 
        fls = struct_data[var_fields[1]][0, 0]  # [0,0] because MATLAB structs become arrays

        if self.in_channels==1:
            amplitude = fls[-1]
            amplitude = self._apply_gaussian_noise(amplitude)
            fls[-1] = amplitude# / amplitude.max(axis=1, keepdims=True)
            main_input_tensor_fls = torch.tensor(fls[-1], dtype=torch.float32).unsqueeze(0)
        else:
            frequency, phase_velocity, amplitude = fls[0], fls[1], fls[-1]
            fls[0] = frequency# / frequency.max(axis=0, keepdims=True)
            fls[1] = phase_velocity# / phase_velocity.max(axis=1, keepdims=True)
            amplitude = amplitude / amplitude.max(axis=1, keepdims=True)
            fls[-1] = self._apply_gaussian_noise(amplitude)
            main_input_tensor_fls = torch.tensor(fls, dtype=torch.float32)
        input_tensors.append(main_input_tensor_fls)

        # Process all additional inputs (remaining variables)
        """
        if len(self.in_instances)>1:
            for jj in range(1, len(self.in_instances)):
                add_data= struct_data[var_fields[jj]][0,0]
                add_tensor = torch.tensor(add_data, dtype=torch.float32).unsqueeze(0)
                if add_tensor.dim() == 1:
                    add_tensor = add_tensor.unsqueeze(0)
                input_tensors.append(add_tensor)
        if self.output_dir is None: #self.test_measured:
            return tuple(input_tensors)

        # Otherwise, load output CSV file
        output_data = pd.read_csv(self.output_files[idx], header=None).values
        output_tensor = torch.tensor(output_data, dtype=torch.float32).squeeze()

        # Return all inputs plus the output
        return *input_tensors, output_tensor

if __name__ == "__main__":
    # Create dataset and dataloader
    in_instances = ['fvs', 'fls', 'x0', 'dx', 'Ch']
    in_channels = 1
    main_input_dir = 'training_data/dataset/input'
    output_dir = 'training_data/dataset/output'
    dataset = MyDataset(in_instances, in_channels, main_input_dir, output_dir)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    train_ratio = 0.8
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Print shapes of the first training sample
    for data in train_dataset:
        # Unpack the data - handle variable number of inputs
        *inputs, output = data

        # Print shapes of all inputs
        print(f"Number of input tensors: {len(inputs)}")
        for i, input_tensor in enumerate(inputs):
            print(f"Input {i} shape: {input_tensor.shape}")

        # Print output shape
        print(f"Output shape: {output.shape}")

        # You can also specifically identify main_input and add_inputs if needed
        main_input_fvs = inputs[0]
        main_input_fls = inputs[1]
        add_inputs = inputs[2:] if len(inputs) > 1 else []

        print(f"Main input fvs shape: {main_input_fvs.shape}")
        print(f"Main input fls shape: {main_input_fls.shape}")
        if add_inputs:
            for i, add_input in enumerate(add_inputs):
                print(f"Additional input {i} shape: {add_input.shape}")

        # Just print info for the first sample
        break