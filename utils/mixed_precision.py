from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairModules:
    key: str
    stem: str
    parent_path: str
    u_name: str
    v_name: str
    u_module: nn.Module
    v_module: nn.Module


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    if path == "":
        return model
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _split_parent_and_name(path: str) -> Tuple[str, str]:
    if "." not in path:
        return "", path
    parent, name = path.rsplit(".", 1)
    return parent, name


def _safe_stem(module_name: str) -> Optional[str]:
    if module_name.endswith("_u_proj"):
        return module_name[: -len("_u_proj")]
    if module_name.endswith("_v_proj"):
        return module_name[: -len("_v_proj")]
    return None


def discover_low_rank_pairs(model: nn.Module) -> List[PairModules]:
    named = dict(model.named_modules())
    pairs: List[PairModules] = []
    for path, module in named.items():
        if not isinstance(module, nn.Linear):
            continue
        if not path.endswith("_u_proj"):
            continue
        parent_path, u_name = _split_parent_and_name(path)
        stem = _safe_stem(u_name)
        if stem is None:
            continue
        v_name = f"{stem}_v_proj"
        parent = _get_submodule(model, parent_path)
        if not hasattr(parent, v_name):
            continue
        v_module = getattr(parent, v_name)
        if not isinstance(v_module, nn.Linear):
            continue
        key = f"{parent_path}.{stem}" if parent_path else stem
        pairs.append(
            PairModules(
                key=key,
                stem=stem,
                parent_path=parent_path,
                u_name=u_name,
                v_name=v_name,
                u_module=module,
                v_module=v_module,
            )
        )
    return pairs


def collect_kfac_stats_diagonal(
    model: nn.Module,
    pairs: List[PairModules],
    dataloader,
    device: str = "cuda",
    nsamples: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    model.eval()

    for pair in pairs:
        in_dim = pair.v_module.in_features
        out_dim = pair.u_module.out_features
        stats[pair.key] = {
            "A_diag": torch.zeros(in_dim, dtype=torch.float64, device=device),
            "B_diag": torch.zeros(out_dim, dtype=torch.float64, device=device),
            "count_a": torch.tensor(0.0, dtype=torch.float64, device=device),
            "count_b": torch.tensor(0.0, dtype=torch.float64, device=device),
        }

        def mk_fwd_pre(k):
            def _hook(module, inputs):
                x = inputs[0].detach()
                x_flat = x.float().reshape(-1, x.shape[-1])
                x2 = x_flat.pow(2).sum(dim=0)
                stats[k]["A_diag"] += x2
                stats[k]["count_a"] += x2.new_tensor(x_flat.shape[0], dtype=torch.float64)

            return _hook

        def mk_bwd(k):
            def _hook(module, grad_input, grad_output):
                g = grad_output[0].detach()
                g_flat = g.float().reshape(-1, g.shape[-1])
                g2 = g_flat.pow(2).sum(dim=0)
                stats[k]["B_diag"] += g2
                stats[k]["count_b"] += g2.new_tensor(g_flat.shape[0], dtype=torch.float64)

            return _hook

        handles.append(pair.v_module.register_forward_pre_hook(mk_fwd_pre(pair.key)))
        handles.append(pair.u_module.register_full_backward_hook(mk_bwd(pair.key)))

    for idx, batch in enumerate(dataloader):
        if idx >= nsamples:
            break
        input_ids, labels = batch[0].to(device), batch[1].to(device)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()

    for h in handles:
        h.remove()

    for key, v in stats.items():
        ca = max(v["count_a"].item(), 1.0)
        cb = max(v["count_b"].item(), 1.0)
        v["A_diag"] = (v["A_diag"] / ca).float().cpu()
        v["B_diag"] = (v["B_diag"] / cb).float().cpu()
        del v["count_a"]
        del v["count_b"]

    model.zero_grad(set_to_none=True)
    model.config.use_cache = use_cache
    return stats


def compute_component_importance(
    pairs: List[PairModules], stats: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.key not in stats:
            continue
        A_diag = stats[pair.key]["A_diag"].to(pair.v_module.weight.device, dtype=torch.float32)
        B_diag = stats[pair.key]["B_diag"].to(pair.u_module.weight.device, dtype=torch.float32)
        U = pair.u_module.weight.data.float()
        V = pair.v_module.weight.data.float()
        u_term = (U.pow(2) * B_diag[:, None]).sum(dim=0)
        v_term = (V.pow(2) * A_diag[None, :]).sum(dim=1)
        score = (u_term * v_term).clamp_min(1e-12)
        out[pair.key] = score.detach().cpu()
    return out


def _quant_noise_proxy(bits: int) -> float:
    if bits <= 1:
        return 1.0
    q = (2 ** (bits - 1)) - 1
    return 1.0 / float(q * q)


def solve_budgeted_topk(
    pairs: List[PairModules],
    importance: Dict[str, torch.Tensor],
    low_bit: int = 4,
    high_bit: int = 8,
    avg_bit: float = 4.5,
) -> Dict[str, torch.Tensor]:
    if high_bit <= low_bit:
        raise ValueError("high_bit must be larger than low_bit")

    items = []
    total_params = 0.0
    for pair in pairs:
        score = importance.get(pair.key, None)
        if score is None:
            continue
        out_dim, rank = pair.u_module.weight.shape
        _, in_dim = pair.v_module.weight.shape
        params_per_component = float(out_dim + in_dim)
        total_params += params_per_component * float(rank)
        for i in range(rank):
            value = float(score[i].item()) * (_quant_noise_proxy(low_bit) - _quant_noise_proxy(high_bit))
            delta_cost = params_per_component * float(high_bit - low_bit)
            density = value / max(delta_cost, 1e-12)
            items.append((density, value, delta_cost, pair.key, i))

    target_avg = min(max(avg_bit, float(low_bit)), float(high_bit))
    extra_budget = total_params * (target_avg - float(low_bit))

    items.sort(key=lambda x: x[0], reverse=True)
    selected = set()
    used = 0.0
    for _, value, delta_cost, key, idx in items:
        if value <= 0:
            continue
        if used + delta_cost <= extra_budget + 1e-9:
            selected.add((key, idx))
            used += delta_cost

    alloc: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        rank = pair.u_module.weight.shape[1]
        mask = torch.zeros(rank, dtype=torch.bool)
        for i in range(rank):
            if (pair.key, i) in selected:
                mask[i] = True
        alloc[pair.key] = mask
    return alloc


class TwoPathLowRankLinear(nn.Module):
    def __init__(
        self,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        high_idx: torch.Tensor,
        low_idx: torch.Tensor,
        high_bit: int = 8,
        low_bit: int = 4,
    ):
        super().__init__()
        self.high_bit = high_bit
        self.low_bit = low_bit
        self.out_features = u_weight.shape[0]
        self.register_buffer("high_idx", high_idx.to(torch.long), persistent=False)
        self.register_buffer("low_idx", low_idx.to(torch.long), persistent=False)

        if self.high_idx.numel() > 0:
            uh, vh = u_weight[:, self.high_idx], v_weight[self.high_idx, :]
            self.register_buffer("uh_q", self._q_per_row(uh, high_bit)[0], persistent=True)
            self.register_buffer("uh_s", self._q_per_row(uh, high_bit)[1], persistent=True)
            self.register_buffer("vh_q", self._q_per_row(vh, high_bit)[0], persistent=True)
            self.register_buffer("vh_s", self._q_per_row(vh, high_bit)[1], persistent=True)
        else:
            self.register_buffer("uh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("uh_s", torch.empty(0), persistent=True)
            self.register_buffer("vh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vh_s", torch.empty(0), persistent=True)

        if self.low_idx.numel() > 0:
            ul, vl = u_weight[:, self.low_idx], v_weight[self.low_idx, :]
            self.register_buffer("ul_q", self._q_per_row(ul, low_bit)[0], persistent=True)
            self.register_buffer("ul_s", self._q_per_row(ul, low_bit)[1], persistent=True)
            self.register_buffer("vl_q", self._q_per_row(vl, low_bit)[0], persistent=True)
            self.register_buffer("vl_s", self._q_per_row(vl, low_bit)[1], persistent=True)
        else:
            self.register_buffer("ul_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("ul_s", torch.empty(0), persistent=True)
            self.register_buffer("vl_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vl_s", torch.empty(0), persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().float(), persistent=True)

    @staticmethod
    def _q_per_row(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        qmax = float((2 ** (bits - 1)) - 1)
        w = w.float()
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).cpu()

    @staticmethod
    def _deq_per_row(q: torch.Tensor, s: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if q.numel() == 0:
            return torch.empty(0, device=device, dtype=dtype)
        qf = q.to(device=device, dtype=torch.float32)
        sf = s.to(device=device, dtype=torch.float32).unsqueeze(1)
        return (qf * sf).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        dtype = x.dtype
        dev = x.device
        if self.uh_q.numel() > 0:
            vh = self._deq_per_row(self.vh_q, self.vh_s, dev, dtype)
            uh = self._deq_per_row(self.uh_q, self.uh_s, dev, dtype)
            zh = F.linear(x, vh)
            out_h = F.linear(zh, uh)
            out = out_h if out is None else out + out_h
        if self.ul_q.numel() > 0:
            vl = self._deq_per_row(self.vl_q, self.vl_s, dev, dtype)
            ul = self._deq_per_row(self.ul_q, self.ul_s, dev, dtype)
            zl = F.linear(x, vl)
            out_l = F.linear(zl, ul)
            out = out_l if out is None else out + out_l
        if out is None:
            out = torch.zeros((*x.shape[:-1], self.out_features), device=dev, dtype=dtype)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=dtype)
        return out


def apply_two_path_quantization(
    model: nn.Module,
    pairs: List[PairModules],
    alloc: Dict[str, torch.Tensor],
    high_bit: int = 8,
    low_bit: int = 4,
):
    for pair in pairs:
        if pair.key not in alloc:
            continue
        high_mask = alloc[pair.key]
        rank = high_mask.numel()
        high_idx = torch.where(high_mask)[0]
        low_idx = torch.where(~high_mask)[0]
        if high_idx.numel() == 0 and low_idx.numel() == 0 and rank > 0:
            low_idx = torch.arange(rank)

        mp = TwoPathLowRankLinear(
            u_weight=pair.u_module.weight.data,
            v_weight=pair.v_module.weight.data,
            bias=pair.u_module.bias.data if pair.u_module.bias is not None else None,
            high_idx=high_idx,
            low_idx=low_idx,
            high_bit=high_bit,
            low_bit=low_bit,
        )
        parent = _get_submodule(model, pair.parent_path)
        setattr(parent, f"{pair.stem}_mp_proj", mp)
