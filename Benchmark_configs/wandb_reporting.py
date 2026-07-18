"""Utilities for restricting outbound experiment metrics sent to W&B."""

WANDB_METRIC_ALLOWLIST = frozenset(
    {
        "compression_flops_clients",
        "decompression_flops_clients",
        "round_flops",
        "local_training_flops_round",
        "aggregation_flops_round",
        "evaluation_flops_round",
        "compression_flops_server",
        "decompression_flops_server",
        "acc_servers_highest",
        "overall_traffic",
        "upload_traffic",
        "download_traffic",
        "round",
        "client_id",
        "upload_sparsity_mean",
        "download_sparsity_mean",
    }
)


def filter_wandb_metrics(metrics):
    """Return a new W&B payload containing only allowlisted experiment metrics."""
    return {
        key: value
        for key, value in metrics.items()
        if key in WANDB_METRIC_ALLOWLIST
    }


def log_wandb_metrics(wandb_module, metrics, **kwargs):
    """Log only allowlisted metrics to W&B, skipping empty outbound payloads."""
    wandb_payload = filter_wandb_metrics(metrics)
    if not wandb_payload:
        return False
    wandb_module.log(wandb_payload, **kwargs)
    return True
