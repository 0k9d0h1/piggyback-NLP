import torch.nn as nn
import torch
import math
import avalanche.models as am

from modnets.layers import ElementWiseLinear, ElementWiseEmbedding, MultiTaskClassifier, PretrainingMultiTaskClassifier
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin
from typing import List, Optional, Tuple, Union
from avalanche.benchmarks.scenarios import CLExperience
from torch import Tensor


class BertEmbeddings(am.MultiTaskModule):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, mask_embedding):
        super().__init__()
        self.mask_embedding = mask_embedding
        if self.mask_embedding:
            self.word_embeddings = ElementWiseEmbedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.position_embeddings = ElementWiseEmbedding(
                config.max_position_embeddings, config.hidden_size)
            self.token_type_embeddings = ElementWiseEmbedding(
                config.type_vocab_size, config.hidden_size)
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size)
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(
        self,
        input_ids,
        token_type_ids,
        task_label,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        return self.forward_single_task(input_ids, token_type_ids, task_label, position_ids, inputs_embeds, past_key_values_length)

    def forward_single_task(self,
                            input_ids,
                            token_type_ids,
                            task_label,
                            position_ids: Optional[torch.LongTensor] = None,
                            inputs_embeds: Optional[torch.FloatTensor] = None,
                            past_key_values_length: int = 0,) -> torch.Tensor:
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:,
                                             past_key_values_length: seq_length + past_key_values_length]

        if self.mask_embedding:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids, task_label)
            token_type_embeddings = self.token_type_embeddings(
                token_type_ids, task_label)

            embeddings = inputs_embeds + token_type_embeddings
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(
                    position_ids, task_label)
                embeddings += position_embeddings
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + token_type_embeddings
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(am.MultiTaskModule):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = ElementWiseLinear(config.hidden_size, self.all_head_size)
        self.key = ElementWiseLinear(config.hidden_size, self.all_head_size)
        self.value = ElementWiseLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor],
                task_label,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,):

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        task_label,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states, task_label)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states, task_label))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states, task_label))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self.key(hidden_states, task_label))
            value_layer = self.transpose_for_scores(
                self.value(hidden_states, task_label))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(
                self.key(hidden_states, task_label))
            value_layer = self.transpose_for_scores(
                self.value(hidden_states, task_label))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.dense = ElementWiseLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        return self.forward_single_task(hidden_states, input_tensor, task_label)

    def forward_single_task(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(am.MultiTaskModule):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(
            config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor],
            task_label,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,):

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        task_label,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            task_label,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(
            self_outputs[0], hidden_states, task_label)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.dense = ElementWiseLinear(
            config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward_single_task(self, hidden_states: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.dense = ElementWiseLinear(
            config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        return self.forward_single_task(hidden_states, input_tensor, task_label)

    def forward_single_task(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        task_label,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        task_label,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            task_label,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output, task_label)
        layer_output = self.output(
            intermediate_output, attention_output, task_label)
        outputs = (layer_output,) + outputs

        return outputs


class BertEncoder(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config)
                                   for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        task_label,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        task_label,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                task_label,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.dense = ElementWiseLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward_single_task(self, hidden_states: torch.Tensor, task_label) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor, task_label)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.dense = PretrainingMultiTaskClassifier(
            config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward_single_task(self, hidden_states: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = PretrainingMultiTaskClassifier(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward_single_task(self, hidden_states, task_label):
        hidden_states = self.transform(hidden_states, task_label)
        hidden_states = self.decoder(hidden_states, task_label)
        return hidden_states


class BertPreTrainingHeads(am.MultiTaskModule):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = PretrainingMultiTaskClassifier(
            config.hidden_size, 2)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(self, sequence_output, pooled_output, task_label):
        return self.forward_single_task(sequence_output, pooled_output, task_label)

    def forward_single_task(self, sequence_output, pooled_output, task_label):
        prediction_scores = self.predictions(sequence_output, task_label)
        seq_relationship_score = self.seq_relationship(
            pooled_output, task_label)
        return prediction_scores, seq_relationship_score


class BertModel(am.MultiTaskModule):
    def __init__(self, config, mask_embedding, add_pooling_layer=True):
        super().__init__()
        self.mask_embedding = mask_embedding
        self.config = config

        self.embeddings = BertEmbeddings(config, self.mask_embedding)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = torch.float32
    ) -> Tensor:
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (
            1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(self,
                input_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                token_type_ids: Optional[torch.Tensor],
                task_label,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask,
                                        past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        task_label,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(
        #     head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_label=task_label,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            task_label=task_label,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output, task_label) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForPreTraining(am.MultiTaskModule):

    def __init__(self, config, mask_embedding=False):
        super().__init__()
        self.bert = BertModel(config, mask_embedding)
        self.cls = BertPreTrainingHeads(config)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        task_label,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        task_label,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            task_label=task_label,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, task_label)

        return prediction_scores, seq_relationship_score


class BertForEndtask(am.MultiTaskModule):

    def __init__(self, config, num_classes, mask_embedding=False):
        super().__init__()
        self.bert = BertModel(config, mask_embedding)
        self.cls = MultiTaskClassifier(config.hidden_size, num_classes)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(experience)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        task_label,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        task_label,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            task_label=task_label,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        output = self.cls(
            pooled_output, task_label)

        return output
