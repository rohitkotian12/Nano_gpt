import math

import pytest
import torch

from src.attention import CausalSelfAttention, MultiHeadAttention, copy_loop_weights_to_fused
from src.bigram import BigramLanguageModel
from src.model import GPTLanguageModel, build_model
from src.sampling import sample_logits
from configs.config import get_config

V = 11


def make_gpt(**kw):
    args = dict(vocab_size=V, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    args.update(kw)
    return GPTLanguageModel(**args)


@pytest.mark.parametrize("impl", ["fused", "loop"])
def test_gpt_forward_shapes_and_initial_loss(impl):
    torch.manual_seed(0)
    model = make_gpt(attn_impl=impl)
    idx = torch.randint(0, V, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, V)
    # With the 0.02 init the model starts near the uniform distribution: loss ~ ln(V).
    assert abs(loss.item() - math.log(V)) < 0.3


def test_bigram_forward_and_initial_loss():
    torch.manual_seed(0)
    model = BigramLanguageModel(65)
    idx = torch.randint(0, 65, (4, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 8, 65)
    assert loss.item() > math.log(65) * 0.8  # random init is at or above ln(V)
    assert model(idx)[1] is None


@pytest.mark.parametrize("impl", ["fused", "loop"])
def test_causality_future_tokens_do_not_affect_past_logits(impl):
    torch.manual_seed(1)
    model = make_gpt(attn_impl=impl).eval()
    a = torch.randint(0, V, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % V  # change only the last 4 tokens
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


def test_fused_attention_matches_loop_attention():
    torch.manual_seed(2)
    loop = MultiHeadAttention(32, 4, 16, 0.0).eval()
    fused = CausalSelfAttention(32, 4, 16, 0.0).eval()
    copy_loop_weights_to_fused(loop, fused)
    x = torch.randn(2, 9, 32)
    assert torch.allclose(loop(x), fused(x), atol=1e-5)


def test_sequence_longer_than_block_size_raises():
    model = make_gpt()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_n_embd_must_divide_by_heads():
    with pytest.raises(ValueError):
        get_config("tiny", n_embd=30, n_head=4)


def test_build_model_dispatch():
    assert isinstance(build_model(get_config("bigram"), 65), BigramLanguageModel)
    assert isinstance(build_model(get_config("tiny"), 65), GPTLanguageModel)


# -------------------------------------------------------------- generation
@pytest.mark.parametrize("model", [make_gpt(), BigramLanguageModel(V, block_size=8)])
def test_generate_length_and_range(model):
    idx = torch.zeros((2, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=40)  # longer than block_size -> context is cropped
    assert out.shape == (2, 43)
    assert out.min() >= 0 and out.max() < V
    assert torch.equal(out[:, :3], idx)


def test_generate_is_seed_reproducible_and_restores_train_mode():
    model = make_gpt(dropout=0.1)
    model.train()
    idx = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(idx, 20, generator=torch.Generator().manual_seed(5))
    b = model.generate(idx, 20, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert model.training  # generate() must put the model back in train mode


def test_generate_stream_yields_one_token_at_a_time():
    model = make_gpt()
    pieces = list(model.generate_stream(torch.zeros((1, 1), dtype=torch.long), 7))
    assert len(pieces) == 7 and all(p.shape == (1, 1) for p in pieces)


# ----------------------------------------------------------------- sampling
def test_greedy_sampling_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.3]])
    assert sample_logits(logits, temperature=0).item() == 1


def test_top_k_restricts_candidates():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    seen = {sample_logits(logits, top_k=2, generator=torch.Generator().manual_seed(s)).item() for s in range(200)}
    assert seen <= {3, 4}


def test_top_p_restricts_candidates():
    # probabilities ~ [0.64, 0.24, 0.09, 0.03]; top_p=0.7 keeps only the first two tokens
    logits = torch.log(torch.tensor([[0.64, 0.24, 0.09, 0.03]]))
    seen = {sample_logits(logits, top_p=0.7, generator=torch.Generator().manual_seed(s)).item() for s in range(300)}
    assert seen == {0, 1}


def test_top_p_always_keeps_best_token():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    assert sample_logits(logits, top_p=0.01).item() == 0


def test_low_temperature_concentrates_mass():
    logits = torch.tensor([[1.0, 1.5, 1.2]])
    draws = [sample_logits(logits, temperature=0.05, generator=torch.Generator().manual_seed(s)).item() for s in range(100)]
    assert draws.count(1) >= 95
