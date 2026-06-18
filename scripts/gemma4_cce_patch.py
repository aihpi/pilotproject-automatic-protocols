#!/usr/bin/env python3
"""Cut-Cross-Entropy (CCE) forward for ``Gemma4ForConditionalGeneration``.

Computes the LM loss with Apple's **cut-cross-entropy** kernel, without ever
materialising the full ``seq×vocab(262144)`` logits tensor — so long-context training
fits on large-vocab models (gemma-4). CCE is numerically stable and supports Gemma's
logit softcap natively (an earlier liger fused-CE attempt produced NaN gradients on
this model at long seq / bf16 — see memory `oom-fix-and-bf16-gotcha`).

`cut_cross_entropy.linear_cross_entropy(e, c, targets, softcap=…, shift=True, …)`
takes the hidden states `e` ([B,T,H]) and the lm_head weight `c` ([vocab,H]) directly
and returns the scalar loss without ever building the logits. `shift=True` applies the
causal offset (token i predicts i+1); `ignore_index=-100` skips the masked prompt.

Apply with :func:`patch_gemma4_conditional_generation_cce` *before* ``from_pretrained``.
Gate the fused path on *labels present* (so eval is also memory-safe) and move hidden
states to the (tied) lm_head weight's device under ``device_map="auto"`` (CCE reads the
raw weight; no accelerate hook fires).

Vendored verbatim from the `fix/OOM_issues` branch for reuse by the FSDP track
(scripts/train_lora_fsdp.py --cce). Do not diverge from the upstream copy.
"""
from __future__ import annotations

import sys

import torch

from cut_cross_entropy import linear_cross_entropy


def gemma4_conditional_cce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    mm_token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    per_layer_inputs: torch.Tensor | None = None,
    skip_logits: bool | None = None,
    **kwargs,
):
    """Drop-in ``Gemma4ForConditionalGeneration.forward`` using CCE for the loss."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast

    # Replacing .forward loses the stock @can_return_tuple decorator, which strips
    # return_dict; drop it to avoid colliding with the explicit return_dict=True below.
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        labels=labels,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    text_config = self.config.get_text_config()
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    loss = None
    logits = None

    if skip_logits is None:
        skip_logits = labels is not None

    if skip_logits:
        # CCE reads the raw lm_head weight (no device hook); align under device_map=auto.
        # CCE's backward asserts bf16/fp16 embeddings. In QLoRA, prepare_model_for_kbit_
        # training upcasts the final norm output (hidden states) to fp32, so cast e — and
        # the lm_head weight if it was upcast too — to bf16. This matches the all-bf16 path
        # that trains stably; the loss is still accumulated in fp32 inside the CCE kernel.
        w = self.lm_head.weight
        e = kept_hidden_states.to(device=w.device, dtype=torch.bfloat16)
        if w.dtype not in (torch.bfloat16, torch.float16):
            w = w.to(torch.bfloat16)
        # shift=True: token i predicts i+1; ignore_index=-100 skips the masked prompt;
        # softcap matches Gemma's final_logit_softcapping. No full logits are built.
        loss = linear_cross_entropy(
            e,
            w,
            labels.to(e.device),
            ignore_index=-100,
            softcap=final_logit_softcapping,
            reduction="mean",
            shift=True,
        )
    else:
        # Generation / explicit skip_logits=False: stock full-logits path.
        logits = self.lm_head(kept_hidden_states)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        if labels is not None:
            loss = self.loss_function(logits, labels, text_config.vocab_size, **kwargs)

    return Gemma4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        shared_kv_states=outputs.shared_kv_states,
    )


def patch_gemma4_conditional_generation_cce() -> bool:
    """Monkey-patch the wrapper forward with the CCE forward. Call before load."""
    from transformers.models.gemma4 import modeling_gemma4

    modeling_gemma4.Gemma4ForConditionalGeneration.forward = gemma4_conditional_cce_forward
    print(
        "cce: patched Gemma4ForConditionalGeneration.forward (cut-cross-entropy; "
        "full seq×vocab logits not materialised)",
        file=sys.stderr,
    )
    return True
