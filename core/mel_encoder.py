"""FastSpeech related loss."""

import logging

import torch
from utils.util import make_non_pad_mask
from utils.util import make_pad_mask
from core.embedding import PositionalEncoding
from core.embedding import ScaledPositionalEncoding
from core.encoder import Encoder
from core.modules import initialize


class MelEncoder(torch.nn.Module):

    def __init__(self, idim, adim, aheads, eunits, depth, use_scaled_pos_enc, use_masking, normalize_before,
                 concat_after, pointwise_layer, conv_kernel):


        super(MelEncoder, self).__init__()

        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_masking = use_masking

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        input_layer = torch.nn.Conv1d(idim, adim, 1)
        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=depth,
            input_layer=input_layer,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=pointwise_layer,
            positionwise_conv_kernel_size=conv_kernel,
        )



    def forward(self, mel: torch.Tensor, olens: torch.Tensor) -> torch.Tensor:
        mel = mel[:, : max(olens)]  # torch.Size([32, 868, 80]) -> [B, Lmax, odim]
        mel_masks = self._source_mask(
            olens
        )  # (B, Tmax, Tmax) -> torch.Size([32, 868, 868])

        hs, _ = self.encoder(
            mel, mel_masks
        )

        return hs



    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(device=next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float = 1.0, init_dec_alpha: float = 1.0
    ):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)