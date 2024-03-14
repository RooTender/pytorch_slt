import torch

class FFTConvolve(torch.nn.Module):
    r"""
    Convolves inputs along their last dimension using FFT for complex numbers.
    This is the modified implementation of the original module.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`, where
              `N` and `M` are the trailing dimensions of the two inputs. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """

    def __init__(self, mode: str = "full") -> None:
        super().__init__()
        self.mode = mode

    def _apply_convolve_mode(self, conv_result: torch.Tensor, x_length: int, y_length: int, mode: str) -> torch.Tensor:
        valid_convolve_modes = ["full", "valid", "same"]
        if mode == "full":
            return conv_result
        elif mode == "valid":
            target_length = max(x_length, y_length) - min(x_length, y_length) + 1
            start_idx = (conv_result.size(-1) - target_length) // 2
            return conv_result[..., start_idx : start_idx + target_length]
        elif mode == "same":
            start_idx = (conv_result.size(-1) - x_length) // 2
            return conv_result[..., start_idx : start_idx + x_length]
        else:
            raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): First convolution operand, with shape `(..., N)`.
            y (torch.Tensor): Second convolution operand, with shape `(..., M)`
                (leading dimensions must be broadcast-able with those of ``x``).

        Returns:
            torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
            the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.
        """
        n = x.size(-1) + y.size(-1) - 1
        fresult = torch.fft.fft(x, n=n) * torch.fft.fft(y, n=n)
        result = torch.fft.ifft(fresult, n=n)
        return self._apply_convolve_mode(result, x.size(-1), y.size(-1), mode=self.mode)
