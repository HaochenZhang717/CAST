
import matplotlib.pyplot as plt
import torch


results = torch.load("dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples.pth", map_location=torch.device('cpu'))


for  idx in range(10):
    # 选一个样本
    # idx = 0   # 你可以改成想看的 index

    ts = results["all_reals"][idx].squeeze().cpu().numpy()   # (300,)
    label = results["all_labels"][idx].cpu().numpy()         # (300,)

    t = range(len(ts))

    plt.figure(figsize=(12, 4))

    # 画 time series
    plt.plot(t, ts, linewidth=1.5)

    # 用红色阴影标异常
    plt.fill_between(
        t,
        ts.min(),
        ts.max(),
        where=label > 0,
        alpha=0.25
    )

    plt.title(f"Sample {idx}")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()