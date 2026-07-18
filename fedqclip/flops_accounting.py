"""Layer-aware FLOP accounting helpers for FedQClip."""

import torch
import torch.nn as nn


class ForwardFlopCounter:
    """Counts FLOPs for modules actually executed during forward passes."""

    def __init__(self, model):
        self.model = model
        self.flops = 0.0
        self.handles = []

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()

    def register(self):
        self.remove()
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                self.handles.append(module.register_forward_hook(self._module_hook))
            elif module.__class__.__name__ == "BasicBlock":
                # ResNet BasicBlock performs one residual tensor addition in forward().
                self.handles.append(module.register_forward_hook(self._residual_add_hook))
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset(self):
        self.flops = 0.0

    @staticmethod
    def _first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = ForwardFlopCounter._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _pair(value):
        if isinstance(value, tuple):
            return value
        return (value, value)

    def _module_hook(self, module, inputs, output):
        out = self._first_tensor(output)
        if out is None:
            return
        out_numel = out.numel()
        if isinstance(module, nn.Conv2d):
            batch_size = out.shape[0]
            output_channels = out.shape[1]
            output_height = out.shape[2]
            output_width = out.shape[3]
            kernel_height, kernel_width = self._pair(module.kernel_size)
            input_channels_per_group = module.in_channels // module.groups
            self.flops += float(
                2 * batch_size * output_channels * output_height * output_width
                * input_channels_per_group * kernel_height * kernel_width
            )
        elif isinstance(module, nn.Linear):
            batch_size = out.shape[0] if out.dim() > 1 else 1
            self.flops += float(2 * batch_size * module.in_features * module.out_features)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            self.flops += float(4 * out_numel)
        elif isinstance(module, nn.ReLU):
            self.flops += float(out_numel)
        elif isinstance(module, nn.MaxPool2d):
            kernel_height, kernel_width = self._pair(module.kernel_size)
            self.flops += float(out_numel * kernel_height * kernel_width)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            inp = self._first_tensor(inputs)
            in_numel = 0 if inp is None else inp.numel()
            self.flops += float(in_numel + out_numel)

    def _residual_add_hook(self, module, inputs, output):
        out = self._first_tensor(output)
        if out is not None:
            self.flops += float(out.numel())


def trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clipping_flops_for_model(model, scalar_clipping_operations=4):
    flops = 0.0
    for param in model.parameters():
        if param.grad is not None:
            flops += (2 * param.grad.numel()) + scalar_clipping_operations
    return float(flops)


def assert_round_flop_consistency(round_flops, local_training_flops_round,
                                  aggregation_flops_round, evaluation_flops_round,
                                  compression_flops_server,
                                  decompression_flops_clients):
    expected = local_training_flops_round + aggregation_flops_round + evaluation_flops_round
    assert round_flops == expected, (
        "round_flops must equal local_training + aggregation + evaluation FLOPs; "
        f"got {round_flops}, expected {expected}"
    )
    assert compression_flops_server == 0, "server-to-client compression FLOPs must be omitted"
    assert decompression_flops_clients == 0, "server-to-client decompression FLOPs must be omitted"
