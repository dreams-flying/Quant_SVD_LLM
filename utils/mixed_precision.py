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


def collect_kfac_stats_block_b(
    model: nn.Module,
    pairs: List[PairModules],
    dataloader,
    device: str = "cuda",
    nsamples: int = 8,
    block_size: int = 128,
    collect_a_diag: bool = True,
    shrink_lambda: float = 0.1,
    diag_damp: float = 1e-6,
) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    handles = []
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    model.eval()

    for pair in pairs:
        out_dim = pair.u_module.out_features
        nblocks = (out_dim + block_size - 1) // block_size
        stats[pair.key] = {
            "block_size": block_size,
            "out_dim": out_dim,
            "B_blocks": [],
            "count_b": 0.0,
            "A_diag": torch.zeros(pair.v_module.in_features, dtype=torch.float64),
            "count_a": 0.0,
        }
        for bi in range(nblocks):
            s = bi * block_size
            e = min((bi + 1) * block_size, out_dim)
            stats[pair.key]["B_blocks"].append(torch.zeros((e - s, e - s), dtype=torch.float64))

        def mk_fwd_pre(k):
            def _hook(module, inputs):
                if not collect_a_diag:
                    return
                x = inputs[0].detach().float()
                x_flat = x.reshape(-1, x.shape[-1])
                stats[k]["A_diag"] += x_flat.pow(2).sum(dim=0).to(dtype=torch.float64).cpu()
                stats[k]["count_a"] += float(x_flat.shape[0])

            return _hook

        def mk_bwd(k):
            def _hook(module, grad_input, grad_output):
                g = grad_output[0].detach().float()
                g_flat = g.reshape(-1, g.shape[-1])
                stats[k]["count_b"] += float(g_flat.shape[0])
                bsz = int(stats[k]["block_size"])
                for bi, blk in enumerate(stats[k]["B_blocks"]):
                    s = bi * bsz
                    e = min((bi + 1) * bsz, g_flat.shape[-1])
                    gb = g_flat[:, s:e]
                    cov = gb.transpose(0, 1).matmul(gb).to(dtype=torch.float64).cpu()
                    stats[k]["B_blocks"][bi].add_(cov)

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
        cb = max(float(v["count_b"]), 1.0)
        for bi in range(len(v["B_blocks"])):
            blk = (v["B_blocks"][bi] / cb).float()
            if shrink_lambda > 0:
                blk_diag = torch.diag(torch.diag(blk))
                blk = (1.0 - shrink_lambda) * blk + shrink_lambda * blk_diag
            if diag_damp > 0:
                blk = blk + diag_damp * torch.eye(blk.shape[0], dtype=blk.dtype)
            v["B_blocks"][bi] = blk
        ca = max(float(v["count_a"]), 1.0)
        if collect_a_diag:
            v["A_diag"] = (v["A_diag"] / ca).float()
        else:
            v["A_diag"] = None
        del v["count_b"]
        del v["count_a"]

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


def compute_component_importance_block_b(
    pairs: List[PairModules],
    stats: Dict[str, Dict[str, object]],
    a_mode: str = "adaptive",
    adaptive_alpha: Optional[float] = None,
    adaptive_tau: float = 0.5,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.key not in stats:
            continue
        U = pair.u_module.weight.data.float()
        V = pair.v_module.weight.data.float()
        dev = U.device
        rank = U.shape[1]
        b_term = torch.zeros(rank, device=dev, dtype=torch.float32)
        blk_size = int(stats[pair.key]["block_size"])
        B_blocks: List[torch.Tensor] = stats[pair.key]["B_blocks"]  # type: ignore[assignment]
        for bi, B_blk_cpu in enumerate(B_blocks):
            s = bi * blk_size
            e = min((bi + 1) * blk_size, U.shape[0])
            U_blk = U[s:e, :]
            B_blk = B_blk_cpu.to(device=dev, dtype=torch.float32)
            b_term += (U_blk * (B_blk.matmul(U_blk))).sum(dim=0)

        if a_mode == "identity":
            a_term = V.pow(2).sum(dim=1)
        else:
            A_diag_cpu = stats[pair.key].get("A_diag", None)
            if A_diag_cpu is None:
                raise ValueError("A_diag was not collected but a_mode is not identity")
            A_diag = A_diag_cpu.to(dev, dtype=torch.float32)
            if a_mode == "diag":
                a_weight = A_diag
            elif a_mode == "adaptive":
                mean_a = A_diag.mean().clamp_min(1e-12)
                norm_a = A_diag / mean_a
                if adaptive_alpha is None:
                    cv2 = torch.mean((norm_a - 1.0).pow(2))
                    alpha = (cv2 / (cv2 + adaptive_tau)).clamp(0.0, 1.0)
                else:
                    alpha = torch.tensor(float(adaptive_alpha), device=dev, dtype=torch.float32).clamp(0.0, 1.0)
                a_weight = (1.0 - alpha) + alpha * norm_a
            else:
                raise ValueError(f"Unsupported a_mode: {a_mode}")
            a_term = (V.pow(2) * a_weight[None, :]).sum(dim=1)
        score = (b_term * a_term).clamp_min(1e-12)
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
    sigma_calib: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
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
            if sigma_calib is not None and pair.key in sigma_calib:
                sigma_low = float(sigma_calib[pair.key]["low"][i].item())
                sigma_high = float(sigma_calib[pair.key]["high"][i].item())
                gain = max(sigma_low - sigma_high, 0.0)
            else:
                gain = _quant_noise_proxy(low_bit) - _quant_noise_proxy(high_bit)
            value = float(score[i].item()) * gain
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


def _quantize_vec_symmetric(vec: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = float((2 ** (bits - 1)) - 1)
    if qmax <= 0:
        return torch.zeros_like(vec)
    scale = vec.abs().max().clamp_min(1e-8) / qmax
    q = torch.round(vec / scale).clamp(-qmax, qmax)
    return q * scale


def _rank1_outer_error(u: torch.Tensor, v: torch.Tensor, uq: torch.Tensor, vq: torch.Tensor) -> torch.Tensor:
    uu = torch.dot(u, u)
    vv = torch.dot(v, v)
    uquq = torch.dot(uq, uq)
    vqvq = torch.dot(vq, vq)
    cross = torch.dot(u, uq) * torch.dot(v, vq)
    return (uu * vv + uquq * vqvq - 2.0 * cross).clamp_min(0.0)


def calibrate_component_sigma(
    pairs: List[PairModules],
    low_bit: int = 4,
    high_bit: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    sigma: Dict[str, Dict[str, torch.Tensor]] = {}
    for pair in pairs:
        U = pair.u_module.weight.data.float().cpu()
        V = pair.v_module.weight.data.float().cpu()
        rank = U.shape[1]
        low = torch.zeros(rank, dtype=torch.float32)
        high = torch.zeros(rank, dtype=torch.float32)
        for i in range(rank):
            u = U[:, i]
            v = V[i, :]
            uq_l = _quantize_vec_symmetric(u, low_bit)
            vq_l = _quantize_vec_symmetric(v, low_bit)
            uq_h = _quantize_vec_symmetric(u, high_bit)
            vq_h = _quantize_vec_symmetric(v, high_bit)
            low[i] = _rank1_outer_error(u, v, uq_l, vq_l)
            high[i] = _rank1_outer_error(u, v, uq_h, vq_h)
        sigma[pair.key] = {"low": low, "high": high}
    return sigma


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
        self._runtime_u: Optional[torch.Tensor] = None
        self._runtime_v: Optional[torch.Tensor] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_dtype: Optional[torch.dtype] = None
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

    def _build_runtime_cache(self, device: torch.device, dtype: torch.dtype):
        if (
            self._runtime_u is not None
            and self._runtime_v is not None
            and self._runtime_device == device
            and self._runtime_dtype == dtype
        ):
            return
        parts_u = []
        parts_v = []
        if self.uh_q.numel() > 0:
            parts_u.append(self._deq_per_row(self.uh_q, self.uh_s, device, dtype))
            parts_v.append(self._deq_per_row(self.vh_q, self.vh_s, device, dtype))
        if self.ul_q.numel() > 0:
            parts_u.append(self._deq_per_row(self.ul_q, self.ul_s, device, dtype))
            parts_v.append(self._deq_per_row(self.vl_q, self.vl_s, device, dtype))
        if len(parts_u) == 0:
            self._runtime_u = torch.zeros((self.out_features, 0), device=device, dtype=dtype)
            self._runtime_v = torch.zeros((0, 0), device=device, dtype=dtype)
        elif len(parts_u) == 1:
            self._runtime_u = parts_u[0]
            self._runtime_v = parts_v[0]
        else:
            # Runtime fuse two paths into one low-rank chain: y = [Uh Ul]([Vh;Vl]x)
            self._runtime_u = torch.cat(parts_u, dim=1)
            self._runtime_v = torch.cat(parts_v, dim=0)
        self._runtime_device = device
        self._runtime_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        dev = x.device
        self._build_runtime_cache(dev, dtype)
        if self._runtime_u is None or self._runtime_v is None or self._runtime_v.numel() == 0:
            out = torch.zeros((*x.shape[:-1], self.out_features), device=dev, dtype=dtype)
        else:
            z = F.linear(x, self._runtime_v)
            out = F.linear(z, self._runtime_u)
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
