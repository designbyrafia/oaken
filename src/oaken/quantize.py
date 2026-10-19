
import torch
import torch.nn.functional as F
from typing import Optional


def _e2m1_round_magnitude(m):
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=m.device,
        dtype=m.dtype
    )

    idx = (
        (m > 0.25).long()
        + (m >= 0.75).long()
        + (m > 1.25).long()
        + (m >= 1.75).long()
        + (m > 2.5).long()
        + (m >= 3.5).long()
        + (m > 5.0).long()
    )

    return levels[idx.clamp(max=len(levels) - 1)]


class FP4Quantizer:
    @classmethod
    def quantize_dequantize(cls, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.clone()

        x = tensor.float()
        magnitude = _e2m1_round_magnitude(x.abs().clamp(max=6.0))
        return torch.copysign(magnitude, x).reshape(tensor.shape).to(tensor.dtype)


class FP4E4M3Quantizer:
    @staticmethod
    def quantize_dequantize(
        tensor: torch.Tensor,
        scale_mode: str = "token",
        group_size: int = 32
    ) -> torch.Tensor:
        if scale_mode not in ("token", "group"):
            raise ValueError("scale_mode must be 'token' or 'group'")

        if tensor.numel() == 0:
            return tensor.clone()

        if scale_mode == "group" and group_size <= 0:
            raise ValueError("group_size must be positive")

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        x = tensor.float().reshape(-1, last_dim)

        if scale_mode == "token":
            amax = x.abs().amax(dim=-1, keepdim=True)
            scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
            scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-8)

            normalized = (x / scale).clamp(-6.0, 6.0)
            magnitude = _e2m1_round_magnitude(normalized.abs())
            reconstructed = normalized.sign() * magnitude * scale

        else:
            padding = (group_size - last_dim % group_size) % group_size

            if padding:
                x = F.pad(x, (0, padding))

            padded_dim = x.shape[-1]
            grouped = x.reshape(x.shape[0], padded_dim // group_size, group_size)

            amax = grouped.abs().amax(dim=-1, keepdim=True)
            scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
            scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-8)

            normalized = (grouped / scale).clamp(-6.0, 6.0)
            magnitude = _e2m1_round_magnitude(normalized.abs())
            reconstructed = (normalized.sign() * magnitude * scale).reshape(
                x.shape[0], padded_dim
            )[:, :last_dim]

        return reconstructed.reshape(original_shape).to(original_dtype)


class FP5E4M3Quantizer:
    """Software-defined FP5-like E2M2 magnitude codebook plus sign and E4M3 scale."""

    @staticmethod
    def quantize_dequantize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.clone()

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        last_dim = original_shape[-1]

        x = tensor.float().reshape(-1, last_dim)

        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
        scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-8)

        normalized = (x / scale).clamp(-7.0, 7.0)

        levels = torch.tensor(
            [
                0.0, 0.25, 0.5, 0.75,
                1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5,
                4.0, 5.0, 6.0, 7.0
            ],
            device=x.device,
            dtype=x.dtype
        )

        distances = (normalized.abs().unsqueeze(-1) - levels).abs()
        indices = distances.argmin(dim=-1)
        magnitude = levels[indices]

        reconstructed = normalized.sign() * magnitude * scale

        return reconstructed.reshape(original_shape).to(original_dtype)


class OakenQuantizer:
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 5
    FLOAT_TOLERANCE = 1e-6

    @classmethod
    def get_outlier_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lower: float,
        threshold_upper: float
    ):
        outlier_mask = (
            (input_tensor <= threshold_lower)
            | (input_tensor >= threshold_upper)
        )
        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask
        return inlier, outlier, outlier_mask

    @classmethod
    def get_multigroup_threshold(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float]
    ):
        if len(threshold_lowers) != len(threshold_uppers):
            raise ValueError("Threshold lower/upper lists must have equal lengths")

        if len(threshold_lowers) < 2:
            raise ValueError("At least two threshold groups are required")

        group_masks = []
        group_tensors = []
        prev_thr_low, prev_thr_up = None, None

        for idx, (thr_low, thr_up) in enumerate(
            zip(threshold_lowers, threshold_uppers)
        ):
            if idx == len(threshold_lowers) - 1:
                mask = (
                    (input_tensor > prev_thr_low)
                    & (input_tensor < prev_thr_up)
                )

            elif prev_thr_low is not None and prev_thr_up is not None:
                mask = (
                    ((input_tensor > prev_thr_low) & (input_tensor <= thr_low))
                    | ((input_tensor >= thr_up) & (input_tensor < prev_thr_up))
                )

            else:
                mask = (input_tensor <= thr_low) | (input_tensor >= thr_up)

            group_masks.append(mask)
            group_tensors.append(input_tensor * mask)

            prev_thr_low, prev_thr_up = thr_low, thr_up

        return group_tensors, group_masks

    @staticmethod
    def uniform_quantization_threshold(
        tensor,
        bits: int,
        minval,
        maxval
    ):
        # Retained for compatibility with existing Oaken callers.
        rangeval = maxval - minval
        if torch.is_tensor(rangeval):
            safe_range = torch.where(rangeval == 0, torch.ones_like(rangeval), rangeval)
        else:
            safe_range = rangeval if rangeval != 0 else 1.0

        qx = (2 ** bits - 1) / safe_range
        offset = minval * qx
        quantized = torch.round(qx * tensor - offset)
        quantized = torch.nan_to_num(quantized, nan=2 ** bits - 1)
        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()

        return OakenQuantizer.uniform_quantization_threshold(
            tensor, bits, minval, maxval
        )

    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)
        truncated = int16_tensor & 0b1_11111_1110_0000_00
        return truncated.view(torch.float16)


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):
    @classmethod
    def downsample(
        cls,
        input_tensor: torch.Tensor,
        threshold_lowers: list[float],
        threshold_uppers: list[float],
        quantize_outlier: bool = True,
        use_group_shift: bool = True
    ):
        grouped_tensors, masks = cls.get_multigroup_threshold(
            input_tensor,
            threshold_lowers,
            threshold_uppers
        )

        result_tensor = torch.zeros_like(input_tensor).half()

        if quantize_outlier:
            # Innermost group: FP5.
            grouped_tensors[-1] = FP5E4M3Quantizer.quantize_dequantize(
                grouped_tensors[-1]
            )

            # Outer groups: FP5, except the second-to-last outer group, FP4.
            for idx in range(len(threshold_lowers) - 1):
                low = torch.as_tensor(
                    threshold_lowers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                )
                high = torch.as_tensor(
                    threshold_uppers[idx],
                    device=input_tensor.device,
                    dtype=input_tensor.dtype
                )

                group = grouped_tensors[idx]
                positive_mask = group > 0
                negative_mask = group < 0

                positive = group * positive_mask
                negative = group * negative_mask

                if use_group_shift:
                    positive = positive - high
                    negative = negative - low

                shifted = positive * positive_mask + negative * negative_mask

                if idx == len(threshold_lowers) - 2:
                    quantized = FP4E4M3Quantizer.quantize_dequantize(
                        shifted,
                        scale_mode="token"
                    )
                else:
                    quantized = FP5E4M3Quantizer.quantize_dequantize(shifted)

                positive_q = quantized * positive_mask
                negative_q = quantized * negative_mask

                if use_group_shift:
                    positive_q = positive_q + high
                    negative_q = negative_q + low

                grouped_tensors[idx] = positive_q * positive_mask + negative_q * negative_mask

        else:
            # Middle group: FP4; other groups remain unquantized.
            grouped_tensors[-2] = FP4E4M3Quantizer.quantize_dequantize(
                grouped_tensors[-2],
                scale_mode="token"
            )

        for group, mask in zip(grouped_tensors, masks):
            result_tensor += (group * mask).to(result_tensor.dtype)

        val_frac = [
            (torch.count_nonzero(mask) / mask.numel()).item()
            for mask in masks
        ]

        return result_tensor, val_frac, None
