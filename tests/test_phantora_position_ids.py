import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import modeling_flash_attention_utils


def _load_phantora_utils():
    path = Path(__file__).with_name("phantora_transformers_utils.py")
    spec = importlib.util.spec_from_file_location("phantora_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unpacked_position_ids_avoid_payload_dependent_fa_preparation(monkeypatch):
    position_ids = torch.arange(8192, dtype=torch.int64).unsqueeze(0)
    with patch.object(
        torch.Tensor,
        "nonzero",
        side_effect=RuntimeError("numel: integer multiplication overflow"),
    ):
        try:
            modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids(
                position_ids
            )
        except RuntimeError as exc:
            assert "integer multiplication overflow" in str(exc)
        else:
            raise AssertionError("the unpatched FA preparation failure was not reproduced")

    original_is_packed = modeling_flash_attention_utils._is_packed_sequence
    original_prepare = (
        modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids
    )
    had_marker = hasattr(
        modeling_flash_attention_utils,
        "_phantora_unpacked_position_ids",
    )
    try:
        monkeypatch.setenv("PHANTORA", "1")
        phantora_utils = _load_phantora_utils()
        phantora_utils.install_phantora_transformers_unpacked_position_ids(
            modeling_flash_attention_utils
        )

        with patch.object(
            torch.Tensor,
            "nonzero",
            side_effect=RuntimeError("numel: integer multiplication overflow"),
        ):
            (cu_q, cu_k), (max_q, max_k) = (
                modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids(
                    position_ids
                )
            )

        assert cu_q.tolist() == [0, 8192]
        assert cu_k.tolist() == [0, 8192]
        assert (max_q, max_k) == (8192, 8192)
        assert not modeling_flash_attention_utils._is_packed_sequence(position_ids, 1)
    finally:
        modeling_flash_attention_utils._is_packed_sequence = original_is_packed
        modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids = (
            original_prepare
        )
        if not had_marker and hasattr(
            modeling_flash_attention_utils,
            "_phantora_unpacked_position_ids",
        ):
            del modeling_flash_attention_utils._phantora_unpacked_position_ids


def test_unpacked_position_ids_use_one_sequence_per_batch_row(monkeypatch):
    monkeypatch.setenv("PHANTORA", "1")
    phantora_utils = _load_phantora_utils()
    module = type(
        "FlashAttentionUtils",
        (),
        {
            "_is_packed_sequence": staticmethod(lambda *_: True),
            "prepare_fa_kwargs_from_position_ids": staticmethod(lambda *_: None),
        },
    )
    phantora_utils.install_phantora_transformers_unpacked_position_ids(module)

    position_ids = torch.arange(512, dtype=torch.int64).repeat(8, 1)
    (cu_q, cu_k), (max_q, max_k) = module.prepare_fa_kwargs_from_position_ids(
        position_ids
    )

    expected = [index * 512 for index in range(9)]
    assert cu_q.tolist() == expected
    assert cu_k.tolist() == expected
    assert (max_q, max_k) == (512, 512)
    assert not module._is_packed_sequence(position_ids, 8)
