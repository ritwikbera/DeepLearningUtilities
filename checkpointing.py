from torch.utils.checkpoint import checkpoint_sequential
import torch
import pandas as pd
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt 

def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)


def log_mem(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = model(inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def log_mem_cp(model, inp, mem_log=None, exp=None, cp_chunks=3):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = checkpoint_sequential(model, cp_chunks, inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log

def plot_mem(df, exps=None, normalize_call_idx=True, normalize_mem_all=True):

    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all
            df_.mem_all = df_.mem_all // 2 ** 20

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)

        plot.get_figure().savefig('{} checkpointed'.format(exp))


if __name__=='__main__':
    model = resnet18().cuda()
    bs = 128
    input = torch.rand(bs, 3, 224, 224).cuda()
    mem_log = []

    try:
        mem_log.extend(log_mem(model, input, exp='baseline'))
        mem_log.extend(log_mem_cp(model, input, exp='checkpointed'))

    except Exception as e:
        print(f'log_mem failed because of {e}')

    df = pd.DataFrame(mem_log)

    plot_mem(df, exps=['baseline','checkpointed'])
