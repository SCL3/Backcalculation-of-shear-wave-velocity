import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths for the CSV files
train_loss_file = 'training_history/run-.-tag-Loss_Train.csv'  # Replace with your actual file path
val_loss_file = 'training_history/run-.-tag-Loss_Validation.csv'  # Replace with your actual file path

# Read the CSV files
train_loss_data = pd.read_csv(train_loss_file)
val_loss_data = pd.read_csv(val_loss_file)

# Extract the epochs and losses from the DataFrames
epochs_train = train_loss_data.iloc[:, 1]  # Replace index if needed
train_loss = train_loss_data.iloc[:, 2]

epochs_val = val_loss_data.iloc[:, 1]  # Replace index if needed
val_loss = val_loss_data.iloc[:, 2]

def smooth_data(values, alpha=0.6):
    """
    Smooths a sequence of values using exponential moving average.
    
    Args:
        values (list or np.ndarray): The data to smooth.
        alpha (float): Smoothing factor (0 < alpha <= 1). Higher values mean less smoothing.

    Returns:
        np.ndarray: Smoothed values.
    """
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

# Apply smoothing to both training and validation losses
smooth_train_loss = smooth_data(train_loss, alpha=0.6)
smooth_val_loss = smooth_data(val_loss, alpha=0.6)

# Create a figure for plotting
plt.figure(figsize=(7, 5))  # Set figure size (width x height in inches)

# Plot original training and validation loss with less distinct lines
plt.plot(epochs_train, train_loss, 'r-', linewidth=2, alpha=0.5, label='Original Training Loss')
plt.plot(epochs_val, val_loss, 'b-', linewidth=2, alpha=0.5, label='Original Validation Loss')

# # Plot smoothed training and validation loss with more distinct lines
# plt.plot(epochs_train, smooth_train_loss, 'r-', linewidth=2, label='Smoothed Training Loss')
# plt.plot(epochs_val, smooth_val_loss, 'b-', linewidth=2, label='Smoothed Validation Loss')

# Add labels, title, and legend
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=14)
plt.legend(fontsize=12)

# Adjust tick label font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Display grid
# plt.grid()

# Save the figure with a high resolution suitable for publications
plt.tight_layout()
plt.savefig('training_history/Training_Validation_Loss.png', dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig('Training_Validation_Loss.pdf', dpi=300, bbox_inches='tight')  # Save as PDF

# Show the plot
plt.show()
