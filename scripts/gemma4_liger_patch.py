#!/usr/bin/env python3
"""Liger fused-linear-cross-entropy forward for ``Gemma4ForConditionalGeneration``.

``google/gemma-4-31B-it`` loads as the *multimodal wrapper* class
``Gemma4ForConditionalGeneration``, whose stock ``forward`` materialises the full
``seq_len × vocab(262144)`` logits tensor and an fp32 cross-entropy upcast — ~16 GiB
at seq-len 16384, ~32 GiB at 32768 — which OOMs GPU 0 at the very first step.

Liger 0.8.0 ships a fused-linear-CE forward only for the *text* class
(``Gemma4ForCausalLM.causal_forward``); its ``apply_liger_kernel_to_gemma4_text`` does
**not** touch the ConditionalGeneration wrapper that actually runs, so the patch is a
no-op for this OOM. Liger *does* patch the multimodal wrapper for gemma3
(``apply_liger_kernel_to_gemma3`` → ``multimodal_forward``); this module is the gemma4
equivalent: it mirrors the stock ``Gemma4ForConditionalGeneration.forward``
(``transformers/models/gemma4/modeling_gemma4.py``) verbatim except for the
logits+loss block, which it replaces with liger's ``LigerForCausalLMLoss`` so the
``lm_head`` matmul is fused with CE in chunks and the full logits are never built.

Apply with :func:`patch_gemma4_conditional_generation` *before* ``from_pretrained``.

Two deliberate departures from liger's stock forwards, both required here:

* **Eval safety.** Liger gates the fused path on ``self.training``; during
  ``Trainer.evaluate()`` the model is in eval mode, so the stock path would fall back
  to full logits and OOM at long seq-len. We gate on *labels present* instead, so eval
  also takes the fused path (we only need ``eval_loss``). Pair with
  ``prediction_loss_only=True`` so the trainer never requests eval logits.
* **device_map alignment.** The fused kernel reads ``self.lm_head.weight`` directly
  (no accelerate ``AlignDevicesHook`` fires). With ``device_map="auto"`` and
  ``tie_word_embeddings=True`` the tied lm_head/embedding sits on GPU 0 while the last
  hidden state is on the final shard, so we move the hidden states to the weight's
  device before the matmul (autograd handles the cross-device copy on backward).
"""
from __future__ import annotations

import sys

import torch

from liger_kernel.transformers.model.loss_utils import (
    LigerForCausalLMLoss,
    unpack_cross_entropy_result,
)


def gemma4_conditional_lce_forward(
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
    """Drop-in replacement for ``Gemma4ForConditionalGeneration.forward`` with fused
    linear cross-entropy. Signature mirrors the stock forward so the Trainer/PEFT call
    it unchanged; the only added kwarg is ``skip_logits`` (force the fused path on/off).
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast

    # Identical to the stock wrapper forward — run the multimodal model body.
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
    shift_labels = kwargs.pop("shift_labels", None)

    text_config = self.config.get_text_config()
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    loss = None
    logits = None

    # Fuse lm_head + CE whenever a loss is wanted (train AND eval), so the full
    # seq×vocab logits tensor is never materialised. See module docstring.
    if skip_logits is None:
        skip_logits = labels is not None or shift_labels is not None

    if skip_logits:
        # Raw-weight matmul has no device hook: align hidden states to the (tied)
        # lm_head weight's device under device_map="auto".
        kept_hidden_states = kept_hidden_states.to(self.lm_head.weight.device)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=text_config.hidden_size,
            final_logit_softcapping=final_logit_softcapping,
            **kwargs,
        )
        loss = unpack_cross_entropy_result(result)[0]
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


def patch_gemma4_conditional_generation() -> bool:
    """Monkey-patch ``Gemma4ForConditionalGeneration.forward`` with the fused-CE
    forward. Call *before* ``from_pretrained``. Returns True if patched."""
    from transformers.models.gemma4 import modeling_gemma4

    modeling_gemma4.Gemma4ForConditionalGeneration.forward = gemma4_conditional_lce_forward
    print(
        "liger: patched Gemma4ForConditionalGeneration.forward (fused-linear CE; "
        "full seq×vocab logits not materialised)",
        file=sys.stderr,
    )
    return True
