
import torch
from typing import Optional


class OakenQuantizer:
    # In this version every group is stored as FP4 (E2M1), so both values are 4.
    # Anything that reads these constants for bit accounting will now see 4 bits for all groups.
    QUANTIZE_BITS = 4
    OUTLIER_BITS = 4
    FLOAT_TOLERANCE = 1e-6

    # ---------------- FP4 settings ----------------
    # True: outer, middle and inner groups are ALL quantized to FP4, whatever `quantize_outlier` says.
    FORCE_ALL_GROUPS = True
    # "fp16": per-token scale kept in FP16 (recommended for the first run; isolates the FP4 format effect)
    # "e4m3": per-token scale rounded to FP8 E4M3 (NVFP4-like, no second-level scale; needs PyTorch >= 2.1)
    SCALE_MODE = "fp16"

    FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # E2M1 magnitudes (sign is a separate bit)

    @classmethod
    def fp4_round_magnitude(cls, m: torch.Tensor) -> torch.Tensor:
        """Round non-negative values (already divided by the scale, <= 6) to the nearest E2M1 magnitude."""
        levels = m.new_tensor(cls.FP4_LEVELS)
        idx = (
            (m > 0.25).long() + (m >= 0.75).long()
            + (m > 1.25).long() + (m >= 1.75).long()
            + (m > 2.5).long() + (m >= 3.5).long()
            + (m > 5.0).long()
        )
        return levels[idx]

    @classmethod
    def fp4_quantize_dequantize(cls, tensor: torch.Tensor) -> torch.Tensor:
        """FP4 (E2M1) quantize then dequantize, with one scale per token (last dimension).
        Entries that are zero (e.g. outside the current group) stay zero and do not affect the scale."""
        if tensor.numel() == 0:
            return tensor.clone()
        x = tensor.float()
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
        if cls.SCALE_MODE == "e4m3":
            scale = scale.to(torch.float8_e4m3fn).float()
        else:
            scale = scale.half().float()
        scale = scale.clamp_min(1e-8)
        normalized = (x / scale).clamp(-6.0, 6.0)
        magnitude = cls.fp4_round_magnitude(normalized.abs())
        return torch.sign(normalized) * magnitude * scale

    # ---------------- original helpers (unchanged) ----------------
    @classmethod
    def get_outlier_threshold(cls, input_tensor: torch.Tensor, threshold_lower: float, threshold_upper: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outlier_mask = torch.logical_or(input_tensor <= threshold_lower, threshold_upper <= input_tensor)

        outlier = input_tensor * outlier_mask
        inlier = input_tensor * ~outlier_mask

        return inlier, outlier, outlier_mask

    @classmethod
    def get_multigroup_threshold(cls, input_tensor: torch.Tensor, threshold_lowers: list[float], threshold_uppers: list[float]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        group_masks = list()
        group_tensors = list()
        prev_thr_low, prev_thr_up = None, None
        for idx, (thr_low, thr_up) in enumerate(zip(threshold_lowers, threshold_uppers)):
            if idx == len(threshold_lowers) - 1:
                # Inner-most Group
                group_masks.append(mask := torch.logical_and(input_tensor > prev_thr_low, input_tensor < prev_thr_up))
            elif (prev_thr_low is not None) and (prev_thr_up is not None):
                group_masks.append(mask := torch.logical_or(
                    torch.logical_and(prev_thr_low < input_tensor, input_tensor <= thr_low),
                    torch.logical_and(thr_up <= input_tensor, input_tensor < prev_thr_up))
                )
            else:
                # Outer-most Group
                group_masks.append(mask := torch.logical_or(input_tensor <= thr_low, thr_up <= input_tensor))
            prev_thr_low = thr_low
            prev_thr_up = thr_up

            group_tensors.append(input_tensor * mask)

        assert(len(threshold_lowers) == len(threshold_uppers) == len(group_tensors) == len(group_masks))
        return group_tensors, group_masks

    @staticmethod
    def uniform_quantization_threshold(tensor, bits: int, minval: torch.Tensor, maxval: torch.Tensor):
        rangeval = maxval - minval
        qx = (2 ** bits - 1) / rangeval
        offset = minval * qx
        quantized = torch.round(qx * tensor - offset)
        quantized = torch.nan_to_num(quantized, nan=2 ** bits - 1)
        return (quantized + offset) / qx

    @staticmethod
    def uniform_quantization(tensor, bits: int):
        maxval = torch.max(tensor).cpu().item()
        minval = torch.min(tensor).cpu().item()
        return OakenQuantizer.uniform_quantization_threshold(tensor, bits, minval, maxval)

    @classmethod
    def downsample_mantissa(cls, tensor):
        int16_tensor = tensor.view(torch.int16)
        truncated = int16_tensor & 0b1_11111_1110_0000_00
        return truncated.view(torch.float16)


class MultiThresholdTokenwiseQuantizer(OakenQuantizer):
    @classmethod
    def downsample(cls, input_tensor: torch.Tensor, threshold_lowers: list[float], threshold_uppers: list[float],
                   quantize_outlier: bool = False, use_group_shift: bool = True) -> tuple[torch.Tensor, list[float], torch.Tensor]:
        # Group order: index 0 = outer (largest magnitudes), 1 = middle (bulk), last = inner (closest to zero).
        grouped_tensors, masks = cls.get_multigroup_threshold(input_tensor, threshold_lowers, threshold_uppers)
        n_groups = len(threshold_lowers)
        grouped_tensors = [g.float() for g in grouped_tensors]
        result_tensor = torch.zeros_like(input_tensor).half()

        if quantize_outlier or cls.FORCE_ALL_GROUPS:
            # Inner-most group: no group-shift, FP4
            grouped_tensors[-1] = cls.fp4_quantize_dequantize(grouped_tensors[-1])

            # Outer and middle groups: group-shift toward zero, FP4, shift back
            for idx in range(n_groups - 1):
                group = grouped_tensors[idx]
                upper = float(threshold_uppers[idx])
                lower = float(threshold_lowers[idx])
                positive = group > 0
                negative = group < 0

                if use_group_shift:
                    group = torch.where(positive, group - upper,
                                        torch.where(negative, group - lower, torch.zeros_like(group)))

                quantized = cls.fp4_quantize_dequantize(group)

                if use_group_shift:
                    quantized = torch.where(positive, quantized + upper,
                                            torch.where(negative, quantized + lower, torch.zeros_like(quantized)))

                grouped_tensors[idx] = quantized
        else:
            # Original default behaviour (only reached if FORCE_ALL_GROUPS is False): middle group only, no shift
            grouped_tensors[-2] = cls.fp4_quantize_dequantize(grouped_tensors[-2])

        for tensor, mask in zip(grouped_tensors, masks):
            result_tensor += (tensor * mask).to(result_tensor.dtype)
        heat_map = None

        val_frac = [(torch.count_nonzero(mask) / torch.numel(mask)).item() for mask in masks]
        return (result_tensor, val_frac, heat_map)

