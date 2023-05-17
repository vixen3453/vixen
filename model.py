import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_clip import get_my_clip


class Vixen(pl.LightningModule):

    def __init__(self, lm_model='EleutherAI/gpt-j-6B', ckpt=None):
        super().__init__()
        self.save_hyperparameters()

        self.clip_model, self.clip_preprocess = get_my_clip()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.seq_len = 144

        print('loading model')

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.lm_model, cache_dir=self.cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.LM = AutoModelForCausalLM.from_pretrained(self.hparams.lm_model, cache_dir=self.cache_dir)
        for param in self.LM.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(3072, self.LM.transformer.wte.embedding_dim)

        self.prompt_len = {}
        print('done!')

        if 'ckpt' in self.hparams.keys():
            ckpt = torch.load(self.hparams.ckpt)
            self.load_state_dict(ckpt['state_dict'], strict=False)

    def forward(self, x):
        return self.linear(x)

    def run_lm(self, img_embed, prompt, caption=[''], single=False, **gen_hparams):
        # Text to emb
        # sps = ['The differences between the images are as follows: ' + cap for cap in caption]
        sps = [prompt + cap for cap in caption]
        response = self.tokenizer(sps, return_tensors="pt", padding='longest')
        response_soft = self.LM.transformer.wte(response.input_ids.to(self.device))

        if prompt not in self.prompt_len.keys():
            sps1 = [prompt for cap in caption]
            response1 = self.tokenizer(sps1, return_tensors="pt", padding='longest')
            self.prompt_len[prompt] = response1.input_ids.shape[1]

        total_soft = torch.cat([img_embed, response_soft], 1)

        # Pad target and attention mask
        size = self.seq_len if single else self.seq_len * 2
        target = torch.cat([torch.zeros(len(img_embed), size, dtype=response.input_ids.dtype),
                            response.input_ids], 1).to(self.device)
        attention_mask = torch.cat([torch.ones(len(img_embed), size, dtype=response.attention_mask.dtype),
                                    response.attention_mask], 1).to(self.device)

        # # Run through LM and get loss

        input_embeds = total_soft[:, :size + self.prompt_len[prompt] - 1, :]
        log_captions = [
            self.generate_captions(target[i].unsqueeze(0), input_embeds[i].unsqueeze(0), prompt, size,
                                   **gen_hparams)
            for i in range(len(target))]
        return log_captions

    def generate_captions(self, input_ids, input_embeds, prompt, size, **gen_hparams):
        output = self.LM.generate(input_ids[:, :size + self.prompt_len[prompt] - 1],
                                  inputs_embeds=input_embeds,
                                  do_sample=True,
                                  max_new_tokens=30,
                                  repetition_penalty=3.0,
                                  temperature=0.3,
                                  pad_token_id=self.tokenizer.pad_token_id
                                  ).detach().cpu()
        predicted_caption = self.tokenizer.decode(output[0][size:], skip_special_tokens=True)

        return predicted_caption

    def caption(self, img1, img2):
        img_embed1 = self.clip_preprocess(img1).unsqueeze(0)
        img_embed2 = self.clip_preprocess(img2).unsqueeze(0)

        # Extract CLIP features
        img_embed1 = self.clip_model(img_embed1.type(torch.float32).to(0))  # [-1, 3072, 12, 12]
        img_embed2 = self.clip_model(img_embed2.type(torch.float32).to(0))  # [-1, 3072, 12, 12]
        img_embed1 = img_embed1.flatten(2, 3).permute(0, 2, 1)  # [-1, 144, 3072]
        img_embed2 = img_embed2.flatten(2, 3).permute(0, 2, 1)  # [-1, 144, 3072]

        # Linear projection of image features
        joint_embed = self(torch.cat([img_embed1, img_embed2], 1).type(torch.float32))  # [-1, 288, 4096]

        # Run LM
        captions = self.run_lm(joint_embed, 'The differences between the images are as follows: ')

        return captions[0].replace('The differences between the images are as follows: ', '')
