import time
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from transformers.modeling_outputs import BaseModelOutput
from utils.misc import freeze_params, get_logger
from utils.loss import XentLoss
from .Tokenizer import GlossTokenizer_G2T, TextTokenizer
import math


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size=1024, debug=False, hidden_size=256, num_layers=2, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM'):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, src_feats, src_lens, max_len, hidden=None):
        """
        Args:
            - src_feats: (batch_size, max_src_len, 512)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_emb)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_len)

        return rnn_outputs, packed_emb.sorted_indices


class AdaptiveFusion(nn.Module):
    """
    adaptive Fusion Mechanism
    """

    def __init__(self, input_size_1=512, input_size_2=512, output_siz=2, bias=False):
        """
        adaptive Fusion instead of normal add
        :param input_size_1:
        :param input_size_2:
        :param output_siz:
        :param bias:
        """
        super(AdaptiveFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight_input_1 = nn.Linear(input_size_1, output_siz, bias=bias)
        self.weight_input_2 = nn.Linear(input_size_2, output_siz, bias=bias)
        self.layer_norm = nn.LayerNorm(input_size_1, eps=1e-5)

    def forward(self, input_1, input_2):
        fm_sigmoid = self.sigmoid(self.weight_input_1(input_1) + self.weight_input_2(input_2))
        lambda1 = fm_sigmoid.clone().detach()[:, :, 0].unsqueeze(-1)
        lambda2 = fm_sigmoid.clone().detach()[:, :, 1].unsqueeze(-1)

        fused_output = input_1 + input_2 + torch.mul(lambda1, input_1) + torch.mul(lambda2, input_2)
        fused_output = self.layer_norm(fused_output)
        return fused_output


class AdaptiveMask(nn.Module):
    """
    DDEM v3
    """

    def __init__(self, input_size=512, output_size=512, dropout=0.1):
        """
        AF module
        :param input_size: dimensionality of the input.
        :param output_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(AdaptiveMask, self).__init__()
        self.lstm = BiLSTMLayer(input_size=input_size, hidden_size=output_size, dropout=dropout)
        self.linear = nn.Linear(output_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-5)  # eps=1e-5

    def forward(self, input_tensor, input_len, k=2, mask=None):
        lstm_o, sorted_indices = self.lstm(input_tensor, input_len, input_tensor.shape[1])
        list_out = self.softmax(self.linear(lstm_o).squeeze(-1))
        values, indices = list_out.topk(k, dim=-1, largest=False, sorted=False)
        lstm_o = self.layer_norm(lstm_o)

        # update mask
        if mask is not None:
            sgn_mask_copy = mask.clone().detach()
            for b in range(input_tensor.shape[0]):
                sgn_mask_copy[b, indices[b]] = False
            return indices, lstm_o, sgn_mask_copy
        else:
            return indices, lstm_o


class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type, cfg, task) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task = task
        self.input_type = input_type
        assert self.input_type in ['gloss', 'feature']
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])

        if 'pretrained_model_name_or_path' in cfg:
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
            self.model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                **cfg.get('overwrite_cfg', {})
            )
        elif 'model_config' in cfg:
            self.logger.info('Train translation network from scratch using config={}'.format(cfg['model_config']))
            config = MBartConfig.from_pretrained(cfg['model_config'])
            for k, v in cfg.get('overwrite_cfg', {}).items():
                setattr(config, k, v)
                self.logger.info('Overwrite {}={}'.format(k, v))
            if cfg['TextTokenizer'].get('level', 'sentencepiece') == 'word':
                setattr(config, 'vocab_size', len(self.text_tokenizer.id2token))
                self.logger.info('Vocab_size {}'.format(config.vocab_size))
            self.model = MBartForConditionalGeneration(config=config)

            if 'pretrained_pe' in cfg:
                pe = torch.load(cfg['pretrained_pe']['pe_file'], map_location='cpu')
                self.logger.info('Load pretrained positional embedding from ', cfg['pretrained_pe']['pe_file'])
                with torch.no_grad():
                    self.model.model.encoder.embed_positions.weight = torch.nn.parameter.Parameter(
                        pe['model.encoder.embed_positions.weight'])
                    self.model.model.decoder.embed_positions.weight = torch.nn.parameter.Parameter(
                        pe['model.decoder.embed_positions.weight'])
                if cfg['pretrained_pe']['freeze']:
                    self.logger.info('Set positional embedding frozen')
                    freeze_params(self.model.model.encoder.embed_positions)
                    freeze_params(self.model.model.decoder.embed_positions)
                else:
                    self.logger.info('Set positional embedding trainable')
        else:
            raise ValueError

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=cfg['label_smoothing'])
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))

        if self.task in ['S2T', 'G2T'] and 'pretrained_model_name_or_path' in cfg:
            # in both S2T or G2T, we need gloss_tokenizer and gloss_embedding
            self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])
            self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
            # debug
            self.gls_eos = cfg.get('gls_eos', 'gls')  # gls or txt
        elif self.task == 'S2T_glsfree':
            self.gls_eos = None
            self.gloss_tokenizer, self.gloss_embedding = None, None
        elif 'pretrained_model_name_or_path' not in cfg:
            self.gls_eos = 'txt'
            self.gloss_tokenizer, self.gloss_embedding = None, None
        else:
            raise ValueError

        if cfg.get('from_scratch', False) == True:
            self.model.init_weights()
            self.logger.info('Train Translation Network from scratch!')
        if cfg.get('freeze_txt_embed', False) == True:
            freeze_params(self.model.model.shared)
            self.logger.info('Set txt embedding frozen')

        if 'load_ckpt' in cfg:
            self.load_from_pretrained_ckpt(cfg['load_ckpt'])

        self.encoder = self.model.get_encoder()
        self.adaptive_mask_module1 = AdaptiveMask(input_size=1024, output_size=1024, dropout=0.1)
        self.adaptive_mask_module2 = AdaptiveMask(input_size=1024, output_size=1024, dropout=0.1)
        self.af = AdaptiveFusion(input_size_1=1024, input_size_2=1024)

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        logger = get_logger()
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k, v in checkpoint.items():
            if 'translation_network' in k:
                load_dict[k.replace('translation_network.', '')] = v

        self.load_state_dict(load_dict, strict=False)
        logger.info('Load Translation network from pretrained ckpt {}'.format(pretrained_ckpt))

    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        gloss_embedding = torch.nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.model.config.d_model,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>'])
        self.logger.info('gloss2embed_file ' + gloss2embed_file)
        if from_scratch:
            self.logger.info('Train Gloss Embedding from scratch')
            assert freeze == False
        else:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            self.logger.info('Initialize gloss embedding from {}'.format(gloss2embed_file))
            with torch.no_grad():
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        assert gls in gls2embed, gls
                        gloss_embedding.weight[id_, :] = gls2embed[gls]
                    else:
                        self.logger.info('{} not in gls2embed train from scratch'.format(gls))

        if freeze:
            freeze_params(gloss_embedding)
            self.logger.info('Set gloss embedding frozen')
        return gloss_embedding

    def prepare_gloss_inputs(self, input_ids):
        input_emb = self.gloss_embedding(input_ids) * self.input_embed_scale
        return input_emb

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss_embedding=None, gloss_lengths=None):
        if self.task == 'S2T_glsfree':
            suffix_len = 0
            suffix_embedding = None
        else:
            if self.gls_eos == 'gls':
                suffix_embedding = [self.gloss_embedding.weight[self.gloss_tokenizer.convert_tokens_to_ids('</s>'), :]]
            else:
                suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index, :]]
            if self.task in ['S2T', 'G2T'] and self.gloss_embedding:
                if self.gls_eos == 'gls':
                    src_lang_code_embedding = self.gloss_embedding.weight[ \
                                              self.gloss_tokenizer.convert_tokens_to_ids(self.gloss_tokenizer.src_lang),
                                              :]  # to-debug
                else:
                    src_lang_id = self.text_tokenizer.pruneids[
                        30]  # self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
                    assert src_lang_id == 31
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                suffix_embedding.append(src_lang_code_embedding)
            suffix_len = len(suffix_embedding)
            suffix_embedding = torch.stack(suffix_embedding, dim=0)

        max_length = torch.max(input_lengths) + suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long,
                                     device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]
            if 'gloss+feature' in self.input_type:
                valid_feature = torch.cat(
                    [gloss_embedding[ii, :gloss_lengths[ii], :], feature[:valid_len - gloss_lengths[ii], :]],
                    dim=0)
            else:
                valid_feature = feature[:valid_len, :]  # t,D
            if suffix_embedding != None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0)  # t+2, D
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0] < max_length:
                pad_len = max_length - feature_w_suffix.shape[0]
                padding = torch.zeros([pad_len, feature_w_suffix.shape[1]],
                                      dtype=feature_w_suffix.dtype, device=feature_w_suffix.device)
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0)  # t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len + suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0) * self.input_embed_scale,  # B,T,D
            'attention_mask': attention_mask  # attention_mask
        }
        return transformer_inputs

    def forward(self, **kwargs):
        if self.input_type == 'gloss':
            input_ids = kwargs.pop('input_ids')
            kwargs['inputs_embeds'] = self.prepare_gloss_inputs(input_ids)
        elif self.input_type == 'feature':
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            # quick fix
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
            kwargs = {**kwargs, **new_kwargs}
        else:
            raise ValueError

        _, lstm_1, am_mask1 = self.adaptive_mask_module1(
            input_tensor=kwargs['inputs_embeds'], input_len=input_lengths + 2, k=2, mask=kwargs['attention_mask']
        )
        kwargs['attention_mask'] = am_mask1

        prior_encoder_output_dict = self.encoder(**kwargs, return_dict=True)

        _, lstm_o, am_mask2 = self.adaptive_mask_module2(
            input_tensor=prior_encoder_output_dict['last_hidden_state'], input_len=input_lengths + 2, k=2, mask=kwargs['attention_mask']
        )
        prior_encoder_output_dict['last_hidden_state'] = self.af(prior_encoder_output_dict['last_hidden_state'],
                                                                 lstm_o)
        kwargs['attention_mask'] = am_mask2

        output_dict = self.model(**kwargs, return_dict=True, encoder_outputs=prior_encoder_output_dict)
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob, targets=kwargs['labels'])

        lstm_o = BaseModelOutput(last_hidden_state=lstm_o, hidden_states=None, attentions=None)
        lstm_o_output_dict = self.model(**kwargs, return_dict=True, encoder_outputs=lstm_o)
        lstm_o_log_prob = torch.nn.functional.log_softmax(lstm_o_output_dict['logits'], dim=-1)  # B, T, L
        lstm_o_batch_loss_sum = self.translation_loss_fun(log_probs=lstm_o_log_prob, targets=kwargs['labels'])
        output_dict['translation_loss'] = (batch_loss_sum * 5.0 + lstm_o_batch_loss_sum) / log_prob.shape[0]

        kwargs["encoder_outputs"] = prior_encoder_output_dict
        output_dict['transformer_inputs'] = kwargs  # for later use (decoding)

        return output_dict

    def generate(self,
                 input_ids=None, attention_mask=None,  # decoder_input_ids,
                 inputs_embeds=None, input_lengths=None, encoder_outputs=None,
                 num_beams=5, max_length=100, length_penalty=1, **kwargs):
        assert attention_mask != None
        assert encoder_outputs is not None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones([batch_size, 1], dtype=torch.long,
                                       device=attention_mask.device) * self.text_tokenizer.sos_index
        assert inputs_embeds != None and attention_mask != None

        output_dict = self.model.generate(
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,  # same with forward
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length,
            return_dict_in_generate=True)
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict
