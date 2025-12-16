from torch import nn

def fc_block(in_s, out_s, batch_norm=True, dropout=0.0, softmax=False):
    return nn.Sequential(
        *([] if dropout == 0.0 else [nn.Dropout1d(dropout)]),
        *([nn.BatchNorm1d(in_s)] if batch_norm else []),
        nn.Linear(in_s, out_s),
        nn.Softmax(1) if softmax else nn.LeakyReLU(),
    )


def conv_block(
    in_ch,
    out_ch,
    ks,
    bn=True,
    pool=2,
    dropout: float = 0.0,
    stride: int = 1,
    act: str = "leaky_relu",
    padding_mode: str = 'replicate',
    padding: int | None = None,
    **act_kwargs,
):
    if padding is None:
        padding = ks // 2
    norm = [{"kind": "batch_norm", "num_features": in_ch}] if bn else []
    dl = [{"kind": "dropout", "p": dropout}] if dropout > 0 else []
    pool = [{"kind": "pool", "kernel_size": pool, "stride": pool}] if pool > 1 else []
    return [
        *dl,
        *norm,
        {
            "kind": "conv",
            "kernel_size": ks,
            "stride": stride,
            "padding": padding,
            "in_channels": in_ch,
            "out_channels": out_ch,
            "padding_mode": padding_mode,
        },
        *pool,
        {"kind": act, **act_kwargs},
    ]


def conv_out(
    l_in,
    kernel_size: int = 3,
    padding: int = 0,
    stride: int = 1,
    dilation: int = 1,
    **kwargs,
):
    return (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def pool_out(
    l_in: int,
    kernel_size: int,
    stride: int | None = None,
    dilation: int = 1,
    padding: int = 0,
    **kwargs,
):
    if stride is None:
        stride = kernel_size
    return (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def layer_from_cfg(kind: str | None = None, **kwargs):
    assert kind is not None
    if kind == "conv":
        return nn.Conv1d(**kwargs)
    if kind == "relu":
        return nn.ReLU(**kwargs)
    if kind == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    if kind == "gelu":
        return nn.GELU(**kwargs)
    if kind == "softmax":
        return nn.Softmax(**kwargs)
    if kind == "batch_norm":
        return nn.BatchNorm1d(**kwargs)
    if kind == "dropout":
        return nn.Dropout1d(**kwargs)
    if kind == "pool":
        return nn.AvgPool1d(**kwargs)

    raise ValueError(f"Unknown layer type: {kind}")


def size_after(l_in: int, cfgs: list[dict]):
    l = l_in
    for c in cfgs:
        kind = c["kind"]
        if kind == "conv":
            l = conv_out(l, **c)
        elif kind == "pool":
            l = pool_out(l, **c)
        elif kind in {"relu", "gelu", "leaky_relu", "batch_norm", "dropout", "softmax"}:
            pass
        else:
            raise ValueError(f"Unknown layer type: {kind}")

    return l


def last_out_channels(cfgs: list[dict]):
    for c in reversed(cfgs):
        if "out_channels" in c:
            return c["out_channels"]

    raise ValueError("No layer with out_channels in config")
