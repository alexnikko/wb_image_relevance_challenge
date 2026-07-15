import torch


def step_optimizer(parameters, optimizer, max_norm, scaler=None):
    """Clip unscaled gradients, then perform one optimizer step."""
    parameters = list(parameters)
    if scaler is not None:
        scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()

    return grad_norm
