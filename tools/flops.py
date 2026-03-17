import torch
import training.networks as edm_networks

def compute_model_flops_and_params(model, forward_args=None, forward_kwargs=None):
    total_flops = 0
    handles = []

    def conv2d_hook(module, inputs, output):
        nonlocal total_flops
        w = getattr(module, 'weight', None)
        if w is None:
            return
        n = output.shape[0]
        cout = output.shape[1]
        hout = output.shape[2]
        wout = output.shape[3]
        cin = w.shape[1]
        kh = w.shape[2]
        kw = w.shape[3]
        total_flops += int(n * cout * hout * wout * cin * kh * kw * 2)

    def linear_hook(module, inputs, output):
        nonlocal total_flops
        w = getattr(module, 'weight', None)
        if w is None:
            return
        x = inputs[0]
        n = x.shape[0] if x.dim() > 1 else 1
        in_features = w.shape[1]
        out_features = w.shape[0]
        total_flops += int(n * in_features * out_features * 2)

    def unet_block_attention_hook(module, inputs, output):
        nonlocal total_flops
        if getattr(module, 'num_heads', 0):
            n = output.shape[0]
            h = output.shape[2]
            w = output.shape[3]
            q = h * w
            num_heads = module.num_heads
            c_per_head = module.out_channels // num_heads
            n_heads_total = n * num_heads
            total_flops += int(4 * n_heads_total * c_per_head * q * q)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, edm_networks.Conv2d):
            handles.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, torch.nn.Linear) or isinstance(m, edm_networks.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, edm_networks.UNetBlock):
            handles.append(m.register_forward_hook(unet_block_attention_hook))

    with torch.no_grad():
        args = () if forward_args is None else forward_args
        kwargs = {} if forward_kwargs is None else forward_kwargs
        model(*args, **kwargs)

    for h in handles:
        h.remove()

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    params_mb = params_bytes / (1024 ** 2)
    return total_flops, params_mb