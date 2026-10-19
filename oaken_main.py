
from functools import partial
import json
from statistics import mean

from src.oaken.quantize import *
import pandas as pd


def multi_group_oaken_main(args, model, tokenizer, device, runner):
    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

    n_quant_group = quantizer_stat["n_quant_group"]
    n_layer = len(model.get_decoder().layers)

    sparsity_information = {
        "key": [[0.0 for i in range(n_quant_group)] for j in range(n_layer)],
        "value": [[0.0 for i in range(n_quant_group)] for j in range(n_layer)],
        "counter": [0.0 for j in range(n_layer)]
    }

    def tokenwise_quantize_activation_hook(i, module, input, output):
        # V: 3-bit FP3-like payload + 8-bit E4M3 scale.
        tensor = FP4E4M3Quantizer.quantize_dequantize(
            output, scale_mode="token"
        )
        return tensor.half()

    def channelwise_quantize_activation_hook(i, module, input, output):
        # K: retain Oaken thresholds, group shifting, and INT5 paths.
        # The INT4 path inside downsample() is replaced by FP4 + E4M3.
        tensor, sparsity, heatmap = MultiThresholdTokenwiseQuantizer.downsample(
            output,
            quantizer_stat["key"]["lower_threshold"][i],
            quantizer_stat["key"]["upper_threshold"][i],
            args.quant_outlier,
            use_group_shift=True,
        )

        sparsity_information["key"][i] = [
            sum(x) for x in zip(sparsity_information["key"][i], sparsity)
        ]
        sparsity_information["counter"][i] += 1.0

        return tensor.half()

    for i, decoder in enumerate(model.get_decoder().layers):
        decoder.self_attn.v_proj.register_forward_hook(
            partial(tokenwise_quantize_activation_hook, i)
        )
        decoder.self_attn.k_proj.register_forward_hook(
            partial(channelwise_quantize_activation_hook, i)
        )

    runner(args, model, tokenizer, device)

    key_sparsity = []
    key_sparsity_sum = [0.0 for _ in range(n_quant_group)]

    for i in range(n_layer):
        if sparsity_information["counter"][i] > 0:
            layer_key_sparsity = [
                x / sparsity_information["counter"][i]
                for x in sparsity_information["key"][i]
            ]
        else:
            layer_key_sparsity = [0.0 for _ in range(n_quant_group)]

        key_sparsity.append(layer_key_sparsity)

        print(
            f"Decoder {i} Sparsity: "
            f"Key - {key_sparsity[i]}, Value - plain FP4 + E4M3"
        )

        for idx, item in enumerate(key_sparsity[i]):
            key_sparsity_sum[idx] += item

    print(
        "Total Sparsity: "
        f"Key - {[x / n_layer for x in key_sparsity_sum]}, "
        "Value - not applicable (plain FP4 + E4M3)"
    )


def key_channelwise_value_tokenwise_main(args, model, tokenizer, device, runner):
    sparsity_information = {
        "key": [0.0 for _ in range(len(model.get_decoder().layers))],
        "value": [0.0 for _ in range(len(model.get_decoder().layers))],
        "counter": [0.0 for _ in range(len(model.get_decoder().layers))]
    }

    with open(args.quantizer_path, "r") as f:
        quantizer_stat = json.load(f)

        def tokenwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity = TokenwiseQuantizer.downsample(
                output,
                quantizer_stat["value"]["lower_threshold"][i],
                quantizer_stat["value"]["upper_threshold"][i],
                args.quant_outlier,
            )
            sparsity_information["value"][i] += sparsity
            sparsity_information["counter"][i] += 0.5
            return tensor.half()

        def channelwise_quantize_activation_hook(i, module, input, output):
            tensor, sparsity = ChannelwiseQuantizer.downsample(
                output,
                quantizer_stat["key"]["minval"][i],
                quantizer_stat["key"]["maxval"][i],
                quantizer_stat["key"]["lower_threshold"][i],
                quantizer_stat["key"]["upper_threshold"][i],
                args.quant_outlier,
            )
            sparsity_information["key"][i] += sparsity
            sparsity_information["counter"][i] += 0.5
            return tensor.half()

    if args.model in ["opt", "llama"]:
        for i, decoder in enumerate(model.get_decoder().layers):
            decoder.self_attn.v_proj.register_forward_hook(
                partial(tokenwise_quantize_activation_hook, i)
            )
            decoder.self_attn.k_proj.register_forward_hook(
                partial(channelwise_quantize_activation_hook, i)
            )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    runner(args, model, tokenizer, device)

    key_sparsity = []
    value_sparsity = []

    for i in range(len(model.get_decoder().layers)):
        key_sparsity.append(
            sparsity_information["key"][i]
            / sparsity_information["counter"][i]
        )
        value_sparsity.append(
            sparsity_information["value"][i]
            / sparsity_information["counter"][i]
        )
        print(
            f"Decoder {i} Sparsity: "
            f"Key - {key_sparsity[i]}, Value - {value_sparsity[i]}"
        )

    print(
        f"Total Sparsity: Key - {mean(key_sparsity)}, "
        f"Value - {mean(value_sparsity)}"
    )
