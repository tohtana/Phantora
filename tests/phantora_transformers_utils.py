import os

import torch


def install_phantora_transformers_unpacked_position_ids(
    modeling_flash_attention_utils=None,
) -> None:
    """Keep Transformers FA2 position-id control flow shape-derived in Phantora.

    Phantora's client CUDA kernels are payload-free, so the ordinary
    ``_is_packed_sequence`` predicate and
    ``prepare_fa_kwargs_from_position_ids`` cannot safely inspect CUDA tensor
    contents. For the common unpacked contract, one sequence occupies each
    batch row. This shim makes that layout deterministic and can analytically
    construct its cumulative lengths without ``nonzero``, ``diff``, or ``max``.

    Shape/dtype-only call records cannot recover reset points inside a flattened
    packed sequence. Such a caller needs an explicit packed-layout contract;
    this narrow shim intentionally does not invent one from arbitrary payloads.
    """
    if os.environ.get("PHANTORA") is None:
        return
    if modeling_flash_attention_utils is None:
        from transformers import modeling_flash_attention_utils

    if getattr(
        modeling_flash_attention_utils,
        "_phantora_unpacked_position_ids",
        False,
    ):
        return

    def _is_packed_sequence(position_ids, batch_size):
        del batch_size
        return False

    def _prepare_fa_kwargs_from_position_ids(position_ids):
        if position_ids.dim() != 2:
            raise RuntimeError(
                "Phantora unpacked position_ids must have shape "
                "(batch_size, sequence_length)"
            )
        batch_size, sequence_length = map(int, position_ids.shape)
        if batch_size <= 0 or sequence_length <= 0:
            raise RuntimeError(
                "Phantora unpacked position_ids dimensions must be positive"
            )
        cu_seq_lens = torch.arange(
            0,
            (batch_size + 1) * sequence_length,
            sequence_length,
            dtype=torch.int32,
            device=position_ids.device,
        )
        return (cu_seq_lens, cu_seq_lens), (sequence_length, sequence_length)

    modeling_flash_attention_utils._is_packed_sequence = _is_packed_sequence
    modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids = (
        _prepare_fa_kwargs_from_position_ids
    )
    modeling_flash_attention_utils._phantora_unpacked_position_ids = True
