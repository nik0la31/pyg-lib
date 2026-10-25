import pytest
import torch


def skip_unsupported_dtype(dtype: torch.dtype, device: torch.device) -> None:
    if device.type == 'mps' and dtype == torch.double:
        pytest.skip('MPS does not support float64')
