import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predicted, target):
        mse_loss = torch.mean((predicted - target) ** 2)
        return torch.sqrt(mse_loss)

class RMSE_TV_Loss(nn.Module):
    """
    Classic RMSE + Total Variation (TV) regularization applied only
    to the Vs part (per-layer velocities) of the predicted vector.

    The predicted/target vector is structured like the other losses
    in this file: [h_1, ..., h_nL, Vs_1, ..., Vs_nL, Vs_(nL+1)]
    (nL thicknesses followed by nL+1 velocities, the last one being
    the half-space).

    TV penalizes the sum of absolute differences between consecutive
    layer velocities: |Vs_i+1 - Vs_i|. Unlike a plain smoothing penalty,
    the L1 norm on these differences pushes most differences toward zero
    (merging sub-layers that don't correspond to a real transition) while
    still allowing a few large differences (the real layer boundaries),
    producing a reconstructed profile with sharp steps instead of small
    noisy oscillations.

    lam: weight of the TV term relative to RMSE. Needs empirical tuning
         (start around 0.01-0.1; too large => flattened profile,
         too small => back to the current noisy behavior).
    """
    def __init__(self, lam=0.05):
        super().__init__()
        self.lam = lam

    def forward(self, predicted, target):
        # Terme RMSE standard, identique à RMSELoss
        rmse = torch.sqrt(torch.mean((predicted - target) ** 2))

        # Terme TV calculé uniquement sur la partie Vs de la prédiction
        batch_size, num_elements = predicted.shape
        nL = (num_elements - 1) // 2
        VsPred = predicted[:, nL:]  # shape (batch, nL + 1)

        tv = torch.mean(torch.abs(VsPred[:, 1:] - VsPred[:, :-1]))

        total_loss = rmse + self.lam * tv

        return total_loss

class EMD(nn.Module):
    def __init__(self):
        super(EMD, self).__init__()

    def forward(self, predicted, target):
        # Ensure same length
        assert predicted.shape == target.shape, "Shapes must match"

        # Sort both tensors
        predicted_sorted, _ = torch.sort(predicted)
        target_sorted, _ = torch.sort(target)

        # Compute 1D Wasserstein Distance (EMD)
        wd = torch.mean(torch.abs(predicted_sorted - target_sorted))

        return wd  # or torch.sqrt(wd) if you define it as sqrt(EMD)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predicted, target):
        mse_loss = torch.mean((predicted - target) ** 2)
        return mse_loss

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predicted, target):
        epsilon = 1e-10  # Small value to avoid division by zero
        percentage_error = torch.abs((target - predicted) / (target + epsilon)) * 100
        return torch.mean(percentage_error)

class PiecewiseRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        # Step 1: Differentiable loss (for gradient)
        diff_loss = torch.sqrt(torch.mean((predicted - target) ** 2))

        # Step 2: Compute "correct" physical loss (no gradients)
        with torch.no_grad():
            batch_size, num_elements = target.shape
            nL = (num_elements - 1) // 2

            corrected_losses = []
            for i in range(batch_size):
                hTrue = target[i, :nL]
                VsTrue = target[i, nL:]
                hPredict = predicted[i, :nL]
                VsPredict = predicted[i, nL:]

                zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
                zPredict = torch.cat([torch.zeros(1, device=hPredict.device), torch.cumsum(hPredict, dim=0)])
                zc = torch.unique(torch.cat([zTrue, zPredict])).sort()[0]

                idxTrue = torch.bucketize(zc, zTrue, right=True) - 1
                idxPredict = torch.bucketize(zc, zPredict, right=True) - 1

                idxTrue = torch.clamp(idxTrue, 0, VsTrue.shape[0] - 1)
                idxPredict = torch.clamp(idxPredict, 0, VsPredict.shape[0] - 1)

                VscTrue = VsTrue[idxTrue]
                VscPredict = VsPredict[idxPredict]

                corrected_loss = torch.sqrt(torch.mean((VscTrue - VscPredict) ** 2))
                corrected_losses.append(corrected_loss)

            corrected_loss_value = torch.stack(corrected_losses).mean()

        # Step 3: Replace value (but keep diff_loss graph for backprop)
        # The trick: keep the graph from diff_loss, but replace the *value* with the correct one
        final_loss = diff_loss + (corrected_loss_value - diff_loss).detach()

        return final_loss

class Depth_Weighted_RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        # Step 1: Differentiable loss (for gradient)
        diff_loss = torch.sqrt(torch.mean((predicted - target) ** 2))

        # Step 2: Compute "correct" physical loss (no gradients)
        with torch.no_grad():
            batch_size, num_elements = target.shape
            nL = (num_elements - 1) // 2

            corrected_losses = []
            for i in range(batch_size):
                # hTrue = target[i, :nL]
                # VsTrue = target[i, nL:]
                # hPredict = predicted[i, :nL]
                # VsPredict = predicted[i, nL:]
                #
                # zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
                # zPredict = torch.cat([torch.zeros(1, device=hPredict.device), torch.cumsum(hPredict, dim=0)])
                # zc = torch.unique(torch.cat([zTrue, zPredict])).sort()[0]

                # idxTrue = torch.bucketize(zc, zTrue, right=True) - 1
                # idxPredict = torch.bucketize(zc, zPredict, right=True) - 1
                #
                # idxTrue = torch.clamp(idxTrue, 0, VsTrue.shape[0] - 1)
                # idxPredict = torch.clamp(idxPredict, 0, VsPredict.shape[0] - 1)
                #
                # VscTrue = VsTrue[idxTrue]
                # VscPredict = VsPredict[idxPredict]
                #
                # corrected_loss = torch.sqrt(torch.mean((VscTrue - VscPredict) ** 2))
                # corrected_losses.append(corrected_loss)


                hTrue = target[i, :nL]
                VsTrue = target[i, nL:]
                hPred = predicted[i, :nL]
                VsPred = predicted[i, nL:]

                # Compute depth interfaces
                # zTrue = torch.cat([torch.zeros(1), torch.cumsum(hTrue, dim=0)])
                # zPred = torch.cat([torch.zeros(1), torch.cumsum(hPred, dim=0)])
                # zc = torch.unique(torch.cat([zTrue, zPred])).sort()[0]
                zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
                zPred = torch.cat([torch.zeros(1, device=hPred.device), torch.cumsum(hPred, dim=0)])
                zc = torch.unique(torch.cat([zTrue, zPred])).sort()[0]

                # Common layer indices
                idxTrue = torch.bucketize(zc, zTrue, right=True) - 1
                idxPred = torch.bucketize(zc, zPred, right=True) - 1
                idxTrue = torch.clamp(idxTrue, 0, VsTrue.shape[0] - 1)
                idxPred = torch.clamp(idxPred, 0, VsPred.shape[0] - 1)

                # Common Vs profiles
                VscTrue = VsTrue[idxTrue]
                VscPred = VsPred[idxPred]
                # print(VscTrue)
                # print(VscTrue[:-1])
                # Common layer thickness (between consecutive zc)
                dz = zc[1:] - zc[:-1]

                # Depth-weighted RMSE
                # weighted_loss = torch.sqrt(torch.sum((VscTrue[:-1] - VscPred[:-1]) ** 2 * dz) / torch.sum(dz)
                #                            + (VscTrue[-1] - VscPred[-1]) **2 )
                H_inf = dz[-1]  # virtual thickness for half-space
                weighted_loss = torch.sqrt((torch.sum((VscTrue[:-1] - VscPred[:-1]) ** 2 * dz)
                                        + (VscTrue[-1] - VscPred[-1]) ** 2 * H_inf)  # include half-space
                                                    / (torch.sum(dz) + H_inf)  # normalize by total thickness including half-space
                )

                # Depth-weighted MAE
                # H_inf = 0.33  # virtual thickness for half-space
                #
                # weighted_loss = torch.sqrt(
                #                 (
                #     torch.sum(torch.abs(VscTrue[:-1] - VscPred[:-1]) * dz)
                #     + torch.abs(VscTrue[-1] - VscPred[-1]) * H_inf  # include half-space
                # )
                # / (torch.sum(dz) + H_inf)  # normalize by total thickness
                # )

                # print((VscTrue[-1] - VscPred[-1]) **2)

                corrected_losses.append(weighted_loss)

            corrected_loss_value = torch.stack(corrected_losses).mean()

        # Step 3: Replace value (but keep diff_loss graph for backprop)
        # The trick: keep the graph from diff_loss, but replace the *value* with the correct one
        final_loss = diff_loss + (corrected_loss_value - diff_loss).detach()

        return final_loss

class Depth_Weighted_MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        # Step 1: Differentiable loss (for gradient)
        diff_loss = torch.sqrt(torch.mean((predicted - target) ** 2))

        # Step 2: Compute "correct" physical loss (no gradients)
        with torch.no_grad():
            batch_size, num_elements = target.shape
            nL = (num_elements - 1) // 2

            corrected_losses = []
            for i in range(batch_size):
                # hTrue = target[i, :nL]
                # VsTrue = target[i, nL:]
                # hPredict = predicted[i, :nL]
                # VsPredict = predicted[i, nL:]
                #
                # zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
                # zPredict = torch.cat([torch.zeros(1, device=hPredict.device), torch.cumsum(hPredict, dim=0)])
                # zc = torch.unique(torch.cat([zTrue, zPredict])).sort()[0]

                # idxTrue = torch.bucketize(zc, zTrue, right=True) - 1
                # idxPredict = torch.bucketize(zc, zPredict, right=True) - 1
                #
                # idxTrue = torch.clamp(idxTrue, 0, VsTrue.shape[0] - 1)
                # idxPredict = torch.clamp(idxPredict, 0, VsPredict.shape[0] - 1)
                #
                # VscTrue = VsTrue[idxTrue]
                # VscPredict = VsPredict[idxPredict]
                #
                # corrected_loss = torch.sqrt(torch.mean((VscTrue - VscPredict) ** 2))
                # corrected_losses.append(corrected_loss)


                hTrue = target[i, :nL]
                VsTrue = target[i, nL:]
                hPred = predicted[i, :nL]
                VsPred = predicted[i, nL:]

                # Compute depth interfaces
                # zTrue = torch.cat([torch.zeros(1), torch.cumsum(hTrue, dim=0)])
                # zPred = torch.cat([torch.zeros(1), torch.cumsum(hPred, dim=0)])
                # zc = torch.unique(torch.cat([zTrue, zPred])).sort()[0]
                zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
                zPred = torch.cat([torch.zeros(1, device=hPred.device), torch.cumsum(hPred, dim=0)])
                zc = torch.unique(torch.cat([zTrue, zPred])).sort()[0]

                # Common layer indices
                idxTrue = torch.bucketize(zc, zTrue, right=True) - 1
                idxPred = torch.bucketize(zc, zPred, right=True) - 1
                idxTrue = torch.clamp(idxTrue, 0, VsTrue.shape[0] - 1)
                idxPred = torch.clamp(idxPred, 0, VsPred.shape[0] - 1)

                # Common Vs profiles
                VscTrue = VsTrue[idxTrue]
                VscPred = VsPred[idxPred]
                # print(VscTrue)
                # print(VscTrue[:-1])
                # Common layer thickness (between consecutive zc)
                dz = zc[1:] - zc[:-1]
                # Depth-weighted MAE
                # H_inf = 0.1*zc[-1]  # virtual thickness for half-space, 10% of zmax
                # weighted_loss = (
                #         torch.sum(torch.abs(VscTrue[:-1] - VscPred[:-1]) * dz)
                #         + torch.abs(VscTrue[-1] - VscPred[-1]) * H_inf
                #     ) / (torch.sum(dz) + H_inf)
                weighted_loss = (
                        torch.sum(torch.abs(VscTrue[:-1] - VscPred[:-1]) * dz)
                        + torch.abs(VscTrue[-1] - VscPred[-1]) * (0.1*zc[-1])
                    ) / (1.1*zc[-1])
                corrected_losses.append(weighted_loss)

            corrected_loss_value = torch.stack(corrected_losses).mean()

        # Step 3: Replace value (but keep diff_loss graph for backprop)
        # The trick: keep the graph from diff_loss, but replace the *value* with the correct one
        final_loss = diff_loss + (corrected_loss_value - diff_loss).detach()

        return final_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class DW_MAE_Old(nn.Module):
    def __init__(self, H_inf=0.33, num_interp=100):
        """
        H_inf : virtual thickness of half-space (m)
        num_interp : number of interpolation points for each profile
        """
        super().__init__()
        self.H_inf = H_inf
        self.num_interp = num_interp

    def forward(self, predicted, target):
        batch_size, num_elements = target.shape
        nL = (num_elements - 1) // 2
        losses = []

        for i in range(batch_size):
            hTrue, VsTrue = target[i, :nL], target[i, nL:]
            hPred, VsPred = predicted[i, :nL], predicted[i, nL:]

            # Cumulative depth
            zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
            zPred = torch.cat([torch.zeros(1, device=hPred.device), torch.cumsum(hPred, dim=0)])

            # Common interpolation depth grid (linear)
            z_max = torch.max(zTrue[-1], zPred[-1])
            zc = torch.linspace(0, z_max, self.num_interp, device=hTrue.device)

            # Linear interpolation for true and predicted Vs
            VscTrue = F.interpolate(VsTrue.unsqueeze(0).unsqueeze(0), size=self.num_interp, mode='linear', align_corners=True).squeeze()
            VscPred = F.interpolate(VsPred.unsqueeze(0).unsqueeze(0), size=self.num_interp, mode='linear', align_corners=True).squeeze()

            # Depth weights
            dz = zc[1:] - zc[:-1]

            # Weighted MAE over interpolated points
            weighted_loss = torch.sum(torch.abs(VscTrue[:-1] - VscPred[:-1]) * dz) / torch.sum(dz)

            # Half-space contribution
            weighted_loss = weighted_loss + torch.abs(VsTrue[-1] - VsPred[-1]) * self.H_inf / (torch.sum(dz) + self.H_inf)

            losses.append(weighted_loss)

        return torch.stack(losses).mean()

class AreaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        # Step 1. Simple RMSE (for gradient)
        diff_loss = torch.sqrt(torch.mean((predicted - target) ** 2))

        batch_size, num_elements = target.shape
        nL = (num_elements - 1) // 2

        # Step 2. Area difference (differentiable)
        area_losses = []
        for i in range(batch_size):
            hTrue = target[i, :nL]
            VsTrue = target[i, nL:]
            hPredict = predicted[i, :nL]
            VsPredict = predicted[i, nL:]

            area_true = torch.sum(hTrue * VsTrue)
            area_pred = torch.sum(hPredict * VsPredict)
            area_diff = (area_pred - area_true) ** 2  # or torch.abs(...)
            area_losses.append(area_diff)

        area_loss = torch.stack(area_losses).mean()

        # Step 3. Corrected RMSE by interpolation (no gradient)
        with torch.no_grad():
            corrected_losses = []
            for i in range(batch_size):
                hTrue = target[i, :nL]
                VsTrue = target[i, nL:]
                hPredict = predicted[i, :nL]
                VsPredict = predicted[i, nL:]

                # depth interfaces
                zTrue = torch.cat([torch.zeros(1, device=hTrue.device), torch.cumsum(hTrue, dim=0)])
                zPredict = torch.cat([torch.zeros(1, device=hPredict.device), torch.cumsum(hPredict, dim=0)])
                zc = torch.unique(torch.cat([zTrue, zPredict])).sort()[0]

                # piecewise indices
                idxTrue = torch.bucketize(zc, zTrue, right=True) - 1
                idxPredict = torch.bucketize(zc, zPredict, right=True) - 1
                idxTrue = torch.clamp(idxTrue, 0, VsTrue.shape[0] - 1)
                idxPredict = torch.clamp(idxPredict, 0, VsPredict.shape[0] - 1)

                VscTrue = VsTrue[idxTrue]
                VscPredict = VsPredict[idxPredict]
                corrected_loss = torch.sqrt(torch.mean((VscTrue - VscPredict) ** 2))
                corrected_losses.append(corrected_loss)

            corrected_loss_value = torch.stack(corrected_losses).mean()

        # Step 4. Combine — numerically correct loss, simple gradient
        total_loss = (diff_loss + area_loss) + (corrected_loss_value - (diff_loss + area_loss)).detach()
        return total_loss

