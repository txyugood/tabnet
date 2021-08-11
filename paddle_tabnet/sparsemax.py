import paddle
from paddle import nn
from paddle.autograd import PyLayer
import paddle.nn.functional as F

import numpy as np


"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, axis=0):
    d = input.shape[axis]
    rho = paddle.arange(1, d + 1, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    rho = paddle.reshape(rho, view)
    perm = list(range(len(view)))
    if axis == -1:
        axis = len(view) - 1
    perm[0] = axis
    perm[axis] = 0

    rho = paddle.transpose(rho, perm)

    return rho


class SparsemaxFunction(PyLayer):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, axis=-1, trainning=True):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        axis : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.axis = axis

        max_val = paddle.max(input, keepdim=True, axis=axis)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, axis=axis)
        output = paddle.clip(input - tau, min=0)
        if trainning:
            ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensor()
        axis = ctx.axis
        grad_input = grad_output.clone()
        # grad_input[output == 0] = 0

        idx = paddle.fluid.layers.where(output == 0)
        grad_input_gather = paddle.gather_nd(grad_input, idx)
        grad_input_gather = 0 - grad_input_gather
        grad_input = paddle.scatter_nd_add(grad_input, idx, grad_input_gather)

        v_hat = paddle.sum(grad_input, axis=axis)
        supp_size = paddle.cast(supp_size, dtype=output.dtype)
        supp_size = paddle.squeeze(supp_size, axis=-1)
        v_hat = v_hat / supp_size

        v_hat = paddle.unsqueeze(v_hat, axis=axis)

        grad_input = paddle.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input

    @staticmethod
    def _threshold_and_support(input, axis=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        axis : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """

        input_srt = paddle.sort(input, descending=True, axis=axis)
        input_cumsum = paddle.cumsum(input_srt, axis=axis) - 1
        rhos = _make_ix_like(input, axis)
        support = rhos * input_srt > input_cumsum
        support = paddle.cast(support, dtype='int64')

        support_size = paddle.sum(support, axis=axis)

        # tau = input_cumsum.gather(dim, support_size - 1)

        support_size_range = paddle.arange(1, input_cumsum.shape[0] + 1)
        support_size_range = paddle.unsqueeze(support_size_range, axis=-1)
        # support_size_range.stop_gradient = True
        support_size_nd = paddle.unsqueeze(support_size, axis=-1)
        support_size_nd = paddle.concat([support_size_range, support_size_nd], axis=-1)
        tau = paddle.gather_nd(input_cumsum, support_size_nd-1)
        tau = paddle.unsqueeze(tau, axis=1)
        support_size = paddle.unsqueeze(support_size, axis=-1)

        tau /= paddle.cast(support_size, input.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Layer):

    def __init__(self, axis=-1):
        self.axis = axis
        super(Sparsemax, self).__init__()

    def forward(self, input, training):
        return sparsemax(input, self.axis, training)


class Entmax15Function(PyLayer):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(PyLayer):
    """ A highly optimized equivalent of lambda x: Entmax15([x, 0]) """

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


class Entmax15(nn.Layer):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)


# Credits were lost...
# def _make_ix_like(input, dim=0):
#     d = input.size(dim)
#     rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
#     view = [1] * input.dim()
#     view[0] = -1
#     return rho.view(view).transpose(0, dim)
#
#
# def _threshold_and_support(input, dim=0):
#     """Sparsemax building block: compute the threshold
#     Args:
#         input: any dimension
#         dim: dimension along which to apply the sparsemax
#     Returns:
#         the threshold value
#     """
#
#     input_srt, _ = torch.sort(input, descending=True, dim=dim)
#     input_cumsum = input_srt.cumsum(dim) - 1
#     rhos = _make_ix_like(input, dim)
#     support = rhos * input_srt > input_cumsum
#
#     support_size = support.sum(dim=dim).unsqueeze(dim)
#     tau = input_cumsum.gather(dim, support_size - 1)
#     tau /= support_size.to(input.dtype)
#     return tau, support_size
#
#
# class SparsemaxFunction(Function):
#
#     @staticmethod
#     def forward(ctx, input, dim=0):
#         """sparsemax: normalizing sparse transform (a la softmax)
#         Parameters:
#             input (Tensor): any shape
#             dim: dimension along which to apply sparsemax
#         Returns:
#             output (Tensor): same shape as input
#         """
#         ctx.dim = dim
#         max_val, _ = input.max(dim=dim, keepdim=True)
#         input -= max_val  # same numerical stability trick as for softmax
#         tau, supp_size = _threshold_and_support(input, dim=dim)
#         output = torch.clamp(input - tau, min=0)
#         ctx.save_for_backward(supp_size, output)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         supp_size, output = ctx.saved_tensors
#         dim = ctx.dim
#         grad_input = grad_output.clone()
#         grad_input[output == 0] = 0
#
#         v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
#         v_hat = v_hat.unsqueeze(dim)
#         grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
#         return grad_input, None
#
#
# sparsemax = SparsemaxFunction.apply
#
#
# class Sparsemax(nn.Module):
#
#     def __init__(self, dim=0):
#         self.dim = dim
#         super(Sparsemax, self).__init__()
#
#     def forward(self, input):
#         return sparsemax(input, self.dim)
if __name__ == '__main__':
    from paddle.nn import Linear
    loss = Sparsemax(axis=1)
    layer = Linear(100, 10)
    inputs = paddle.unsqueeze(paddle.arange(1, 101), axis=0)
    inputs = paddle.cast(inputs, dtype='float32')
    out = layer(inputs)
    l = loss(out)
    l = paddle.mean(l)
    l.backward()
    pass

