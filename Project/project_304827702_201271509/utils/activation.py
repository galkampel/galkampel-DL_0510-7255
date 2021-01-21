
import torch


class DropoutMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p=0.5, training=True):
        device = input.device
        if training:
            mask = (torch.rand(*input.shape, dtype=torch.float, device=device) > p) / (1 - p)
            # print(f'#non-zeros in mask:\t{mask.nonzero().size(0)}')
            out = input * mask
            mask = mask > 0
            ctx.save_for_backward(mask)
            return out, mask

        mask = torch.ones_like(input, dtype=torch.float, device=device)
        out = input * mask
        ctx.save_for_backward(mask)
        return out, mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        mask = ctx.saved_tensors[0]
        masked_grad_output = grad_output * mask
        return masked_grad_output, None, None


dropout = DropoutMask.apply
