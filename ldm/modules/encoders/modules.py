import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

from ldm.modules.x_transformer import Encoder, TransformerWrapper

import open_clip
from ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SetTransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              use_pos_emb=False)

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)

class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


import numpy as np
import transformers
import hashlib
import zlib
from base64 import urlsafe_b64decode as b64d
import six

MAX_AUG = 50

class TagTokenizer(object):
    def __init__(self, ar_model_path, max_length, device='cuda'):
        self.device = device
        self.max_length = max_length

        # TODO: this is hacky, can we do smth nicer? OmegaConf doesn't seem to allow specifying relative paths.
        import os
        our_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(our_dir, 'tags.config'), 'rb') as f:
          raw_tags = f.read()
        id2tag = zlib.decompress(b64d(raw_tags))
        id2tag = six.ensure_str(id2tag).split(',')
        tag2id = {t: i for i, t in enumerate(id2tag)}
        id2tag = {i: t for t, i in tag2id.items()}

        self.vocab_size = len(id2tag)
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.cache = {}
        self.bos = 2999
        self.eos = 2999
        self.pad = 2999
        self.pad_token_id = self.pad

        vocab_size = len(id2tag)
        self.BOS = vocab_size
        self.EOS = vocab_size + 1
        self.PAD = vocab_size + 2

        self.model = transformers.AutoModelForCausalLM.from_pretrained(ar_model_path).to(self.device)

        BLACKLIST = 'censored,mosaic_censoring,photoshop_(medium),comic,monochrome,translated,greyscale,jpeg_artifacts,hard_translated,bar_censor,lowres,bad_pixiv_id,bad_id,translation_request,transparent_background,realistic,photo_(medium)'.split(',')
        META = 'rating_e,rating_s,rating_q,score_perc_10,score_perc_20,score_perc_40,score_perc_60,score_perc_80,score_perc_90,score_perc_100,adjusted_score_perc_10,adjusted_score_perc_20,adjusted_score_perc_40,adjusted_score_perc_60,adjusted_score_perc_80,adjusted_score_perc_90,adjusted_score_perc_100'.split(',')
        self.BLACKLIST = [[tag2id[t]] for t in BLACKLIST]
        self.META = [tag2id[t] for t in META]

    def token_id(self, token):
        if token in self.tag2id:
          return self.tag2id[token]
        h = self.pad + 1 + (int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16) % (10 ** 9))
        self.cache[h] = token
        return h

    def token(self, token_id):
        if token_id in self.id2tag:
          return self.id2tag[token_id]
        assert token_id in self.cache
        return self.cache[token_id]

    def generate(self, prompt, max_augment=MAX_AUG, seed=None):
      result = []
      ids = [self.BOS]
      index_mapping = []
      for t in prompt:
         result.append(self.token_id(t))
         if t in self.tag2id:
           ids.append(self.tag2id[t])
           index_mapping.append(len(result) - 1)
      if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

      bad_word_ids = [e[0] for e in self.BLACKLIST] + self.META
      bad_word_ids.extend(range(2000, self.BOS - 1))
      bad_word_ids = [[x] for x in set(bad_word_ids)]

      gens = self.model.generate(
          input_ids=torch.tensor(np.array([ids], dtype=np.int32)).to(self.device),
          temperature=1.0,  # !!!
          top_p=0.9,
          do_sample=True,
          min_length=max_augment + 2, max_length=max_augment + 2,
          pad_token_id=self.PAD,
          bos_token_id=self.BOS,
          no_repeat_ngram_size=1,
          bad_words_ids=bad_word_ids).cpu().numpy()[0, 1:-1]
      assert len(gens) >= len(index_mapping)
      for i in range(len(index_mapping)):
          assert result[index_mapping[i]] == gens[i]
      for i in range(len(index_mapping), len(gens)):
          result.append(gens[i])

      return [self.token(x) for x in result]

    def encode_one(self, text, max_length=50, add_special_tokens=True, augment=True, max_augment=MAX_AUG, seed=None, **kwargs):
        text = text.replace(',', ' ').split(' ')
        text = [s for s in text if s]
        if '__NO_AUGMENT__' in text:
          augment = False
          text = [s for s in text if s != '__NO_AUGMENT__']
        print('Called for:', text)
        if len(text) > 0 and augment:
          text = self.generate(text, max_augment=max_augment, seed=seed)
          print('Generated:', text)
        text = [self.token_id(s) for s in text]
        text = text[:max_length]
        if add_special_tokens:
          text.extend([self.pad] * (max_length - len(text)))
        return np.array(text)

    def __call__(self, text, max_length=50, add_special_tokens=True, augment=True, max_augment=MAX_AUG, **kwargs):
        print('Final call with:', augment, max_augment)
        if isinstance(text, list):
            res = np.array([self.encode_one(s, add_special_tokens=add_special_tokens, augment=augment, max_augment=max_augment, **kwargs) for s in text])
        else:
          res = self.encode_one(text, add_special_tokens=add_special_tokens, augment=augment, max_augment=max_augment, **kwargs)
        res = torch.tensor(res)
        return {'input_ids': res}




class WrappedTransformerEmbedder(TransformerEmbedder):
    def __init__(self, device='cuda', max_seq_len=77, ar_model_path='', **kwargs):
        super().__init__(device=device, max_seq_len=max_seq_len, **kwargs)
        self.tokenizer = TagTokenizer(ar_model_path, max_seq_len, device=device)
        self.device = device
        self.max_length = max_seq_len
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, augment=True, **kwargs):
        batch_encoding = self.tokenizer(text, max_length=self.max_length, augment=augment, **kwargs)
        if isinstance(batch_encoding, dict):
            batch_encoding = batch_encoding['input_ids']
        tokens = torch.clamp(batch_encoding, 0, self.tokenizer.pad)
        tokens = batch_encoding.to(self.device)
        z = super().forward(tokens)
        return z

    def encode(self, text):
        return self(text)

