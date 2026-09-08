"""Validation for the quantisation-agnostic arming path (backends/generic.py).

The kernels are covered by test_policy.py. What is new and worth attacking here is the
decision layer: which tensors get mirrored, whether the rebuilt quant config really ends
up bound to slot storage, whether a scheme that hides a per-expert tensor is *refused*
rather than silently corrupted, and whether the layer is left exactly as it was found.

Everything runs on CPU against synthetic stand-ins for vLLM's classes, so it needs no GPU
and no vLLM install -- the point is the logic, not the arithmetic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, "/home/dave/vllm-expert-cache")

from vllm_expert_cache.backends import generic  # noqa: E402

E, S = 16, 4
H, N = 8, 12
FAIL = []


def chk(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL:", msg)
    else:
        print("  ok:", msg)
    return cond


class Log:
    """Collects warnings so a test can assert on the reason a layer was declined."""

    def __init__(self):
        self.msgs = []

    def _rec(self, fmt, *a):
        self.msgs.append(fmt % a if a else fmt)

    warning = info = debug = _rec

    def said(self, needle):
        return any(needle in m for m in self.msgs)


@dataclass
class Desc:
    scale: torch.Tensor | None = None
    alpha_or_gscale: torch.Tensor | None = None
    zp: torch.Tensor | None = None
    bias: torch.Tensor | None = None


@dataclass
class Cfg:
    _a1: Desc
    _a2: Desc
    _w1: Desc
    _w2: Desc


class Experts:
    def __init__(self, cfg):
        self.quant_config = cfg


class Kernel:
    def __init__(self, cfg):
        self.fused_experts = Experts(cfg)


class Layer(torch.nn.Module):
    """Stands in for vLLM's RoutedExperts: per-expert tensors as registered Parameters."""

    def __init__(self, extra=(), n_experts=E):
        super().__init__()
        p = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
        self.w13_weight = p(torch.randn(n_experts, N, H))
        self.w2_weight = p(torch.randn(n_experts, H, N // 2))
        for name, shape in extra:
            setattr(self, name, p(torch.randn(*shape)))
        self.global_num_experts = n_experts
        self.expert_map = None
        self.activation = "silu"
        self.apply_router_weight_on_input = False


class Method:
    """Stands in for a FusedMoEMethodBase subclass.

    `reads` names the layer attributes this scheme's config factory pulls in, mirroring
    the real ones (fp8 reads w13_weight_scale, wNa16 also reads zero points, and so on).
    """

    is_monolithic = False

    def __init__(self, layer, reads=(), hidden=None):
        self.reads = list(reads)
        self.hidden = hidden  # a per-expert tensor the factory returns but layer-binding
                              # cannot reach -- the silent-corruption case
        self.moe_quant_config = self._cfg(layer)
        self.moe_kernel = Kernel(self.moe_quant_config)
        self.applied_with = None

    def _cfg(self, layer):
        got = [getattr(layer, n) for n in self.reads]
        w1 = Desc(scale=got[0] if len(got) > 0 else None,
                  zp=got[2] if len(got) > 2 else None)
        w2 = Desc(scale=got[1] if len(got) > 1 else None,
                  zp=got[3] if len(got) > 3 else None)
        if self.hidden is not None:
            w1.bias = self.hidden
        return Cfg(Desc(), Desc(), w1, w2)

    def get_fused_moe_quant_config(self, layer):
        return self._cfg(layer)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None,
              shared_experts_input=None):
        # Record what the stock path would have seen, which is the whole point: these are
        # the tensors the real fused kernel indexes with topk_ids.
        self.applied_with = dict(
            w13=layer.w13_weight, w2=layer.w2_weight,
            cfg=self.moe_kernel.fused_experts.quant_config,
            ids=topk_ids.clone(), gne=layer.global_num_experts)
        return torch.zeros(1)


def arm(layer, method, log=None):
    log = log or Log()
    return generic.arm(layer, method, log), log


# ---------------------------------------------------------------------------------------

def test_unquantized():
    print("unquantized (weights only):")
    layer = Layer()
    m = Method(layer)
    ok, _ = arm(layer, m)
    chk(ok, "armed")
    chk(set(layer._expert_cache.names) == {"w13_weight", "w2_weight"},
        f"mirrors exactly the two weights, got {sorted(layer._expert_cache.names)}")


def test_int8_like():
    print("int8-like (weights + per-channel scales):")
    extra = [("w13_weight_scale", (E, N, 1)), ("w2_weight_scale", (E, H, 1))]
    layer = Layer(extra)
    m = Method(layer, reads=["w13_weight_scale", "w2_weight_scale"])
    ok, _ = arm(layer, m)
    chk(ok, "armed")
    chk(set(layer._expert_cache.names) ==
        {"w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"},
        "mirrors weights and both scales by discovery alone")


def test_wna16_like_six_tensors():
    print("wNa16-like (weights + scales + zero points = 6 tensors):")
    extra = [("w13_weight_scale", (E, N, 1)), ("w2_weight_scale", (E, H, 1)),
             ("w13_weight_zero_point", (E, N, 1)), ("w2_weight_zero_point", (E, H, 1))]
    layer = Layer(extra)
    m = Method(layer, reads=["w13_weight_scale", "w2_weight_scale",
                             "w13_weight_zero_point", "w2_weight_zero_point"])
    ok, _ = arm(layer, m)
    chk(ok, "armed")
    chk(len(layer._expert_cache.names) == 6, "mirrors all six per-expert tensors")


def test_packed_weights_with_alias():
    """The packed compressed-tensors family (wNa16 int4, w4a4-nvfp4/mxfp4) registers
    `w13_weight_packed` and then aliases it with a plain `layer.w13_weight = ...`, which
    is the name `apply` actually reads. Both names must be rebound, and the shared
    storage must be mirrored ONCE."""
    print("packed weights aliased under a second name:")
    layer = Layer()
    # Re-register under the packed names, then alias exactly as vLLM does.
    for base in ("w13", "w2"):
        t = getattr(layer, f"{base}_weight")
        del layer._parameters[f"{base}_weight"]
        layer.register_parameter(f"{base}_weight_packed", t)
        setattr(layer, f"{base}_weight", t)  # plain attribute alias, same storage
    m = Method(layer)
    ok, log = arm(layer, m)
    chk(ok, f"armed, got: {log.msgs}")
    if not ok:
        return
    cache = layer._expert_cache
    chk(len(cache.slots) == 2, f"mirrored 2 storages, not 4 names; got {len(cache.slots)}")
    chk(set(cache.bound) == {"w13_weight", "w13_weight_packed",
                             "w2_weight", "w2_weight_packed"},
        f"binds all four names, got {sorted(cache.bound)}")
    chk(cache.bound["w13_weight"].data_ptr() == cache.bound["w13_weight_packed"].data_ptr(),
        "alias and packed name share one slot buffer")

    cache.refresh = lambda ids: ids.clone()
    m.apply(layer, torch.randn(1, H), torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int32))
    chk(m.applied_with["w13"].data_ptr() == cache.slots["w13_weight_packed"].data_ptr(),
        "apply read slot storage through the alias")
    chk(getattr(layer, "w13_weight").data_ptr() == getattr(
        layer, "w13_weight_packed").data_ptr(), "both names restored to one storage")


def test_wide_step_reads_through():
    print("a step too wide for the slots falls back to full storage:")
    extra = [("w13_weight_scale", (E, N, 1)), ("w2_weight_scale", (E, H, 1))]
    layer = Layer(extra)
    m = Method(layer, reads=["w13_weight_scale", "w2_weight_scale"])
    ok, _ = arm(layer, m)
    assert ok
    # tokens x top_k > slots, i.e. prefill: must take the stock path untouched.
    wide = torch.zeros(1, S + 1, dtype=torch.int32)
    m.apply(layer, torch.randn(1, H), torch.ones(1, S + 1), wide)
    got = m.applied_with
    chk(got["cfg"]._w1.scale.size(0) == E, "read-through saw full [E] scales")
    chk(got["gne"] == E, "read-through kept the full expert count")
    chk(got["w13"].data_ptr() == layer.w13_weight.data_ptr(),
        "read-through saw the layer's own weights")


def test_slot_binding_reaches_the_config():
    print("a step that fits is bound to slot storage, not full storage:")
    extra = [("w13_weight_scale", (E, N, 1)), ("w2_weight_scale", (E, H, 1))]
    layer = Layer(extra)
    m = Method(layer, reads=["w13_weight_scale", "w2_weight_scale"])
    ok, _ = arm(layer, m)
    assert ok
    cache = layer._expert_cache
    # The kernels are covered by test_policy.py; here only the binding is under test.
    cache.refresh = lambda ids: ids.clone()
    m.apply(layer, torch.randn(1, H), torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int32))
    got = m.applied_with
    chk(got["w13"].data_ptr() == cache.slots["w13_weight"].data_ptr(),
        "apply saw slot weights")
    chk(got["cfg"]._w1.scale.size(0) == S,
        f"apply saw slot-sized scales, got dim0={got['cfg']._w1.scale.size(0)}")
    chk(got["gne"] == S, f"global_num_experts narrowed to slots, got {got['gne']}")


def test_apply_accepts_keyword_call():
    """vLLM calls apply(layer=..., x=..., ...) by keyword, so the wrapper's parameter
    names are part of its contract. Positional-only testing hid a TypeError that killed
    every worker at profile_run."""
    print("the wrapped apply must accept vLLM's keyword call:")
    layer = Layer()
    m = Method(layer)
    ok, _ = arm(layer, m)
    assert ok
    layer._expert_cache.refresh = lambda ids: ids.clone()
    try:
        m.apply(layer=layer, x=torch.randn(1, H), topk_weights=torch.ones(1, 2),
                topk_ids=torch.zeros(1, 2, dtype=torch.int32),
                shared_experts=None, shared_experts_input=None)
        chk(True, "accepted all six arguments by keyword")
    except TypeError as e:
        chk(False, f"keyword call rejected: {e}")


def test_layer_is_restored():
    print("the layer is left exactly as found:")
    extra = [("w13_weight_scale", (E, N, 1)), ("w2_weight_scale", (E, H, 1))]
    layer = Layer(extra)
    m = Method(layer, reads=["w13_weight_scale", "w2_weight_scale"])
    before = {n: getattr(layer, n).data_ptr()
              for n in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")}
    ok, _ = arm(layer, m)
    assert ok
    layer._expert_cache.refresh = lambda ids: ids.clone()
    m.apply(layer, torch.randn(1, H), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32))
    after = {n: getattr(layer, n).data_ptr() for n in before}
    chk(before == after, "every mirrored attribute points back at full storage")
    chk(layer.global_num_experts == E, "global_num_experts restored")
    chk(m.moe_kernel.fused_experts.quant_config._w1.scale.size(0) == E,
        "experts.quant_config restored to the full config")


def test_hidden_per_expert_tensor_is_refused():
    print("a per-expert tensor the factory hides must be REFUSED, not corrupted:")
    layer = Layer()
    hidden = torch.randn(E, N)          # dim0 == E, but not an attribute on the layer
    m = Method(layer, hidden=hidden)
    ok, log = arm(layer, m)
    chk(not ok, "declined")
    chk(log.said("stayed bound to full"), "explained which tensor leaked")
    chk(log.said("_w1.bias"), f"named the field, got: {log.msgs}")


def test_monolithic_is_refused():
    print("a monolithic method must be refused (it routes internally):")
    layer = Layer()
    m = Method(layer)
    m.is_monolithic = True
    ok, log = arm(layer, m)
    chk(not ok, "declined")
    chk(log.said("monolithic"), "explained why")


def test_expert_parallel_is_refused():
    print("expert parallelism must be refused (two id spaces):")
    layer = Layer()
    layer.expert_map = torch.arange(E)
    m = Method(layer)
    ok, log = arm(layer, m)
    chk(not ok, "declined")
    chk(log.said("expert parallelism"), "explained why")


def test_ragged_auxiliary_is_caught_by_the_leak_check():
    """A ragged tensor that the config DOES index per expert cannot be mirrored, so the
    layer must be refused -- but by the leak check, which knows it matters, rather than
    by the alignment rule, which cannot tell bookkeeping from a live scale."""
    print("a ragged per-expert tensor the config uses must still be refused:")
    layer = Layer([("w13_weight_scale", (E, 3))])  # 3 x fp32 = 12 bytes
    m = Method(layer, reads=["w13_weight_scale"])
    ok, log = arm(layer, m)
    chk(not ok, "declined")
    chk(log.said("stayed bound to full"), f"caught by the leak check, got: {log.msgs}")
    chk(log.said("w13_weight_scale"), "reported the skipped tensor by name")


def test_ragged_weight_is_refused():
    """A ragged *weight* is fatal and must be refused outright: the fused kernel is
    handed it directly, so there is no later check that would catch it."""
    print("a per-expert weight that is not a 16-byte multiple must be refused by name:")
    layer = Layer()
    layer.w13_weight = torch.nn.Parameter(torch.randn(E, 3), requires_grad=False)
    m = Method(layer)
    ok, log = arm(layer, m)
    chk(not ok, "declined")
    chk(log.said("16 bytes") or log.said("multiple of 16"),
        f"explained the alignment rule, got: {log.msgs}")
    chk(log.said("w13_weight"), "named the offending tensor")


def test_nothing_to_cache():
    print("a slot budget >= expert count arms nothing:")
    import os
    os.environ["EXPERT_CACHE_SLOTS"] = str(E)
    try:
        layer = Layer()
        ok, _ = arm(layer, Method(layer))
        chk(not ok, "declined without error")
    finally:
        os.environ["EXPERT_CACHE_SLOTS"] = str(S)


if __name__ == "__main__":
    import os
    os.environ.setdefault("EXPERT_CACHE_SLOTS", str(S))
    torch.manual_seed(0)
    for fn in (test_unquantized, test_int8_like, test_wna16_like_six_tensors,
               test_packed_weights_with_alias,
               test_wide_step_reads_through, test_slot_binding_reaches_the_config,
               test_apply_accepts_keyword_call, test_layer_is_restored,
               test_hidden_per_expert_tensor_is_refused,
               test_monolithic_is_refused, test_expert_parallel_is_refused,
               test_ragged_auxiliary_is_caught_by_the_leak_check,
               test_ragged_weight_is_refused, test_nothing_to_cache):
        fn()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all generic-arming checks passed")
