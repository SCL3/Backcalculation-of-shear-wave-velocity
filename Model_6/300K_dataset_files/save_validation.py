import matplotlib
matplotlib.use('Agg')  # non-interactive backend: avoids Tkinter init in DataLoader worker processes on Windows
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import scipy
import scipy.io
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Layered model: column vector 7x1

nLayer = 7 # fixed number of real pavement layers, must be within h_bound
numOut = 2 * nLayer - 1 # concat three values of thickness (except inf) with four values of Vs => 7 discrete values

class VelocityProfilePlotter:

    def __init__(self, nLayer, numOut):

        self.nLayer = nLayer
        self.numOut = numOut

        # Create standard axis vectors
        self.Vph = np.arange(100, 1501, 1).reshape(-1, 1)  # Column vector from 100 to 1500 with step 1
        self.freq = np.arange(100, 5001, 50).reshape(-1, 1)  # Column vector from 100 to 5000 with step 50
        self.vv, self.ff = np.meshgrid(self.Vph.flatten(), self.freq.flatten())

    def create_plot(self, input_np, out_np, target_np=None):
        normalized_input = input_np # / input_np.max(axis=1, keepdims=True)
        # Set up figure
        fig = plt.figure(figsize=(10, 8))
        # Plot input data (left subplot)
        self._create_input_subplot(fig, normalized_input)
        # Plot velocity profile (right subplot)
        self._create_velocity_subplot(fig, out_np, target_np)
        # Finalize layout
        plt.tight_layout()
        return fig

    def _create_input_subplot(self, fig, normalized_input):
        ff=normalized_input[0,:,:]
        ll=normalized_input[1,:,:]
        fls=normalized_input[2,:,:]
        ax = fig.add_subplot(1, 2, 1)
        contour = ax.contourf(ll, ff, fls, levels=14, cmap="jet") #RdBu_r
        cbar = fig.colorbar(contour, ax=ax, pad=0.01)
        ax.invert_yaxis()
        ax.set_xlabel('Phase velocity', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(labelsize=14)

    def _create_velocity_subplot(self, fig, out_np, target_np=None):
        ax = fig.add_subplot(1, 2, 2)
        has_target = target_np is not None
        # Process prediction data
        # predicted_h = np.full(100, 0.01)
        # predicted_h[-1] = np.inf
        predicted_h = np.append(out_np[:self.nLayer - 1], np.inf)
        predicted_Vs = out_np[self.nLayer - 1:self.numOut]
        pred_depths = [0] + np.cumsum(predicted_h[:len(predicted_Vs) - 1]).tolist()
        pred_depths.append(pred_depths[-1] * 100)

        # Plot predicted velocity profile
        ax.step(
            [predicted_Vs[0]] + predicted_Vs.tolist(),
            pred_depths,
            where='post',
            linestyle='-' if has_target else '-',
            color='black',
            linewidth=2 if has_target else 2,
            label="Predicted Vs"
        )

        # If target data is available, add comparison plots
        if has_target:
            self._add_target_plots(ax, target_np, pred_depths)

        # Configure velocity plot
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.2])
        ax.invert_yaxis()
        ax.set_xlabel(r'$V_s$', fontsize=14)
        ax.set_ylabel('Depth z', fontsize=14)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(labelsize=14)

        # Add legend with position based on mode
        if has_target:
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
        else:
            ax.legend(loc='lower right')

    def _add_target_plots(self, ax, target_np, pred_depths):
        # Process target data
        h = np.append(target_np[:self.nLayer - 1], np.inf)
        Vs = target_np[self.nLayer - 1:self.numOut]
        target_depths = [0] + np.cumsum(h[:len(Vs) - 1]).tolist()
        target_depths.append(target_depths[-1] * 100)

        # Plot target velocity profile
        ax.step(
            [Vs[0]] + Vs.tolist(),
            target_depths,
            where='post',
            linestyle='-',
            color='red',
            linewidth=2,
            label="Vs"
        )

def save_prediction(input_fvs, predict, file_id, folder_save_result='validation_results',
                    input_x0=None, input_dx=None, input_Ch=None, target=None):

    create_folder(folder_save_result)

    # Make and save the figure
    plotter = VelocityProfilePlotter(nLayer, numOut)
    fig = plotter.create_plot(input_fvs, predict, target)
    fig_path = os.path.join(folder_save_result, f"{file_id}.jpg")
    print(f"Save result to {fig_path}")
    plt.savefig(fname=fig_path, dpi=150)
    plt.close(fig)

    # Save data files
    input_fvs_path = os.path.join(folder_save_result, f"{file_id}_input_fvs.mat")
    scipy.io.savemat(input_fvs_path, {'fvs': input_fvs})  # data is the fieldname

    input_Ch = np.array(input_Ch)
    input_x0 = np.array(input_x0)
    input_dx = np.array(input_dx)

    if input_x0 is not None:
        add_input_path=os.path.join(folder_save_result, f"{file_id}_input_x0.csv")
        np.savetxt(add_input_path, input_x0*(input_Ch-1)*input_dx, delimiter="\n", fmt='%f')
    if input_dx is not None:
        add_input_path=os.path.join(folder_save_result, f"{file_id}_input_dx.csv")
        np.savetxt(add_input_path, input_dx, delimiter="\n", fmt='%f')

    if input_Ch is not None:
        add_input_path=os.path.join(folder_save_result, f"{file_id}_input_Ch.csv")
        np.savetxt(add_input_path, input_Ch, delimiter="\n", fmt='%f')

    predict_path = os.path.join(folder_save_result, f"{file_id}_predict.csv")
    np.savetxt(predict_path, predict, delimiter="\n", fmt='%f')

    if target is not None:
        target_path = os.path.join(folder_save_result, f"{file_id}_output.csv")
        np.savetxt(target_path, target, delimiter="\n", fmt='%f')

    # Clean up
