# -*- coding: utf-8 -*-
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch
from deneb.models.estimators.estimator_base import Estimator
from deneb.modules.feedforward import FeedForward
from deneb.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors
import open_clip
import math

try:
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DenebEstimator(Estimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(Estimator.ModelConfig):
        switch_prob: float = 0.0

    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Estimator:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()

        if self.hparams.encoder_model != "LASER":
            self.layer = (
                int(self.hparams.layer)
                if self.hparams.layer != "mix"
                else self.hparams.layer
            )

            self.scalar_mix = (
                ScalarMixWithDropout(
                    mixture_size=self.encoder.num_layers,
                    dropout=self.hparams.scalar_mix_dropout,
                    do_layer_norm=True,
                )
                if self.layer == "mix" and self.hparams.pool != "default"
                else None
            )

        input_emb_sz = self.encoder.output_units * 2*3
        
        self.ff = torch.nn.Sequential(
            *[
                FeedForward(
                    in_dim=input_emb_sz,
                    hidden_sizes=self.hparams.hidden_sizes,
                    activations=self.hparams.activations,
                    dropout=self.hparams.dropout,
                    final_activation=(
                        self.hparams.final_activation
                        if hasattr(
                            self.hparams, "final_activation"
                        )  # compatability with older checkpoints!
                        else "Sigmoid"
                    ),
                ),
                torch.nn.Sigmoid(),
            ]
        )

        emb_dim = self.encoder.output_units
        trm_layer1 = torch.nn.TransformerEncoderLayer(d_model=emb_dim*2, nhead=8, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(trm_layer1, num_layers=3)
        trm_layer2 = torch.nn.TransformerEncoderLayer(d_model=emb_dim*2, nhead=8, batch_first=True)
        self.transformer2 = torch.nn.TransformerEncoder(trm_layer2, num_layers=3)
        self.upsampler = torch.nn.Linear(emb_dim, emb_dim, dtype=torch.float32)
        self.upsampler2 = torch.nn.Linear(emb_dim, emb_dim, dtype=torch.float32)

        self.pos_encoder = PositionalEncoding(d_model=emb_dim*2, dropout=0.1, max_len=1000)
        self.pos_encoder2 = PositionalEncoding(d_model=emb_dim*2, dropout=0.1, max_len=1000)
        self.cls_token1 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim*2))
        torch.nn.init.normal_(self.cls_token1, std=1e-6)
        self.cls_token2 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim*2))
        torch.nn.init.normal_(self.cls_token2, std=1e-6)

        self.pacs_clip, _, self.pacs_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self.pacs_clip = self.pacs_clip.cuda().float()
        self.pacs_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        pacscheckpoint = torch.load(
            "experiments/openClip_ViT-L-14.pth"
        )["state_dict"]

        self.pacs_clip.load_state_dict(pacscheckpoint, strict=True)
        for param in self.pacs_clip.parameters():
            param.requires_grad = False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """ Sets different Learning rates for different parameter groups. """
        optimizer = self._build_optimizer(self.parameters())
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]
    

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = collate_tensors(sample)
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = [self.encoder.prepare_sample(ref) for ref in sample["refs"]]
        
        inputs = {
            "mt_inputs": mt_inputs,
            "ref_inputs": ref_inputs,
            "refs": sample["refs"],
            "mt": sample["mt"],
            "imgs": sample["img"]
        }

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def masked_global_average_pooling(self, input_tensor, mask):
        mask = mask.logical_not() # mask[x] = input[x] is not pad
        mask_expanded = mask.unsqueeze(-1).expand_as(input_tensor).float()
        input_tensor_masked = input_tensor * mask_expanded
        num_elements = mask.sum(dim=1,keepdim=True).float() # TODO: チェック
        output_tensor = input_tensor_masked.sum(dim=1) / num_elements
        return output_tensor

    def forward(
        self,
        refs,
        mt,
        ref_inputs,
        mt_inputs,
        imgs: torch.tensor,
        alt_tokens: torch.tensor = None,
        alt_lengths: torch.tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes both Source, MT and Reference and returns a quality score.

        :param src_tokens: SRC sequences [batch_size x src_seq_len]
        :param mt_tokens: MT sequences [batch_size x mt_seq_len]
        :param ref_tokens: REF sequences [batch_size x ref_seq_len]
        :param src_lengths: SRC lengths [batch_size]
        :param mt_lengths: MT lengths [batch_size]
        :param ref_lengths: REF lengths [batch_size]

        :param alt_tokens: Alternative REF sequences [batch_size x alt_seq_len]
        :param alt_lengths: Alternative REF lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """
        mt_tokens, mt_lengths = mt_inputs["tokens"], mt_inputs["lengths"]
        mt_sentemb, mt_sentembs, mt_mask, padding_index = self.get_sentence_embedding(mt_tokens, mt_lengths, pooling=False)
        mt_mask = mt_mask.logical_not()

        ref_sentemb_list = []
        ref_sentembs_list = []
        ref_mask_list = []
        for ref in ref_inputs:
            ref_tokens, ref_lengths = ref["tokens"], ref["lengths"]
            ref_sentemb, ref_sentembs, ref_mask, _ = self.get_sentence_embedding(ref_tokens, ref_lengths, pooling=False)
            ref_mask = ref_mask.logical_not()
            ref_sentemb_list.append(ref_sentemb)
            ref_sentembs_list.append(ref_sentembs)
            ref_mask_list.append(ref_mask)

        mt_pacs = self.pacs_tokenizer(["In the image, " + x for x in mt]).to(
            self.device
        )
        imgs_pacs = torch.cat(
            [self.pacs_preprocess(img).unsqueeze(0) for img in imgs], dim=0
        ).to(self.device)
        with torch.no_grad():
            imgs_pacs = self.pacs_clip.encode_image(imgs_pacs)
            mt_pacs = self.pacs_clip.encode_text(mt_pacs)

        refs_pacs = []
        for ref_list in refs:
            subset = [
                self.pacs_tokenizer("In the image, " + ref).to(self.device)
                for ref in ref_list
            ]
            subset = torch.cat(subset, dim=0)
            with torch.no_grad():
                refs_tensor = self.pacs_clip.encode_text(subset)
            refs_pacs.append(refs_tensor)

        del imgs

        diff_clip = self.upsampler(torch.abs(imgs_pacs - mt_pacs))
        mul_clip = self.upsampler(imgs_pacs * mt_pacs)

        x1 = [torch.cat([diff_clip, mul_clip], dim=1)]
        x2 = []
        x3 = torch.cat([diff_clip, mul_clip], dim=1)

        for ref_sentembs, ref_sentemb, ref_mask, ref_pacs in zip(
            ref_sentembs_list, ref_sentemb_list, ref_mask_list, refs_pacs
        ):
            diff_clip_txt = self.upsampler(torch.abs(ref_pacs - mt_pacs))
            mul_clip_txt = self.upsampler(ref_pacs * mt_pacs)
            diff_roberta = self.upsampler2(torch.abs(mt_sentemb - ref_sentemb))
            mul_roberta = self.upsampler2(mt_sentemb * ref_sentemb)
            x1.extend([torch.cat([diff_clip_txt, mul_clip_txt], dim=1)])
            x2.extend([torch.cat([diff_roberta, mul_roberta], dim=1)])

        x1 = torch.stack(x1, dim=1)
        x2 = torch.stack(x2, dim=1)

        cls_token1 = self.cls_token1.expand(x1.shape[0], -1, -1)
        x1 = torch.cat([cls_token1, x1], dim=1)
        x1 = self.pos_encoder(x1.permute(1, 0, 2))
        x1 = x1.permute(1, 0, 2)  
        x1 = self.transformer(x1)
        x1 = x1[:, 0, :]

        cls_token2 = self.cls_token2.expand(x2.shape[0], -1, -1)
        x2 = torch.cat([cls_token2, x2], dim=1)
        x2 = self.pos_encoder2(x2.permute(1, 0, 2))
        x2 = x2.permute(1, 0, 2)
        x2 = self.transformer2(x2)
        x2 = x2[:, 0, :]

        x = torch.cat([x1, x2, x3],dim=1)
        x = x.flatten(1)
        score = self.ff(x)

        return {"score": score.squeeze()}
