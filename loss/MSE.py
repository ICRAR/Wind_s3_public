import torch
from torch import nn
import torch.nn.functional as F


class MSE_point(nn.Module):
    def __init__(self):
        super().__init__()

    def mse(self, pred: torch.Tensor, label: torch.Tensor):
        # mask label, and apply the mask to pred too
        mask = torch.isnan(label)
        label_masked = torch.where(mask, torch.zeros_like(label), label)
        pred_masked = torch.where(mask, torch.zeros_like(pred), pred)
        # Calculate the mean squared error only for non-NaN values
        non_nan_mask = ~mask
        num_non_nan = non_nan_mask.sum()

        # Ensure there are non-NaN values to avoid division by zero
        if num_non_nan > 0:
            loss = F.mse_loss(pred_masked[non_nan_mask], label_masked[non_nan_mask])
        else:
            loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        return loss

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        return self.mse(pred, label)


class MSE_10_3_point(nn.Module):
    def __init__(self):
        super().__init__()

    def mse10(self, pred: torch.Tensor, label: torch.Tensor):
        # mask label, and apply the mask to pred too
        mask = torch.isnan(label)
        label_masked = torch.where(mask, torch.zeros_like(label), label)
        pred_masked = torch.where(mask, torch.zeros_like(pred), pred)
        # Calculate the mean squared error only for non-NaN values
        non_nan_mask = ~mask
        num_non_nan = non_nan_mask.sum()

        # Ensure there are non-NaN values to avoid division by zero
        if num_non_nan > 0:
            loss = F.mse_loss(pred_masked[non_nan_mask], label_masked[non_nan_mask])
        else:
            loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        # print('label10m loss',loss)
        return loss

    def mse3(self, pred: torch.Tensor, label: torch.Tensor):
        # mask label, and apply the mask to pred too
        mask = torch.isnan(label)
        label_masked = torch.where(mask, torch.zeros_like(label), label)
        pred_masked = torch.where(mask, torch.zeros_like(pred), pred)
        # Calculate the mean squared error only for non-NaN values
        non_nan_mask = ~mask
        num_non_nan = non_nan_mask.sum()

        # Ensure there are non-NaN values to avoid division by zero
        if num_non_nan > 0:
            # Calculate the mean and standard deviation for each variable in label_masked and pred_masked
            label_mean = label_masked.mean(dim=(0, 2), keepdim=True)
            label_std = label_masked.std(dim=(0, 2), unbiased=False, keepdim=True)
            pred_mean = pred_masked.mean(dim=(0, 2), keepdim=True)
            pred_std = pred_masked.std(dim=(0, 2), unbiased=False, keepdim=True)

            # Standardize label_masked using z-score normalization
            standardized_label_masked = (label_masked - label_mean) / label_std

            # Scale standardized_label_masked to match the scale of pred_masked
            scaled_label_masked = standardized_label_masked * pred_std + pred_mean

            loss = F.mse_loss(
                pred_masked[non_nan_mask], scaled_label_masked[non_nan_mask]
            )
        else:
            loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        # print('label3m loss',loss)
        return loss

    def forward(self, pred: torch.Tensor, label: torch.Tensor, label3m: torch.Tensor):
        return self.mse10(pred, label) + self.mse3(pred, label3m)
