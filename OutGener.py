import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import CrossEntropyLoss
from utils.eval import compute_batch

import wandb

from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    get_linear_schedule_with_warmup,
)

class OutGenerModel(LightningModule):
    def __init__(self, args, tokenizer, special_tokens: dict):
        super().__init__()
        self.args = args
        if self.args.model_name == 'BART':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path) 
        else:
            self.model = AutoModelWithLMHead.from_pretrained(args.model_path)     
        self.model_name = args.model_name
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings((len(self.tokenizer)))

        self.special_tokens = special_tokens
        self.delimeter_ids = self.tokenizer.convert_tokens_to_ids(special_tokens['delimeter'])
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.model.config.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.model.config.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.model.config.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.model.config.forced_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.args.train_batch_size * max(1, self.trainer.gpus)
        #ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size)# // ab_size

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.total_steps
        )

        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler
                }
            }

    def forward(self, input_ids, max_length=512):
        output = self.model.generate(input_ids, do_sample=True, max_length=max_length,\
            top_k=0, top_p=0.9, length_penalty = self.args.length_penalty,\
            decoder_start_token_id=self.tokenizer.bos_token_id, bos_token_id = self.tokenizer.bos_token_id,\
            eos_token_id=self.tokenizer.eos_token_id, min_length = self.args.min_length)
        return output

    def training_step(self, batch, batch_idx):

        encoded_batch, attention_mask, encoded_label_batch = batch

        output = self.model(encoded_batch, attention_mask=attention_mask, labels=encoded_label_batch, return_dict=True)

        return output['loss']

    def validation_step(self, batch, batch_idx):
        encoded_batch, attention_mask, encoded_label_batch = batch
        output_loss = self.model(encoded_batch, attention_mask=attention_mask, labels=encoded_label_batch, return_dict=True)['loss']
        output = self(encoded_batch)

        label_num = len(encoded_label_batch[encoded_label_batch != -100].view(-1))
        label_batch, inference_result, key_word_batch = \
            self.post_process_for_validation(encoded_batch, encoded_label_batch, output)
        
        return output_loss, label_num, label_batch, inference_result, key_word_batch

    def validation_epoch_end(self, validation_output) -> None:
        outputs, label_batch, inference_result, key_word_batch = [], [], [], []
        for i1, i2, i3, i4, i5 in validation_output:
            outputs.append((i1, i2))
            label_batch.extend(i3)
            inference_result.extend(i4)
            key_word_batch.extend(i5)
        total_loss = 0.0
        total_batch_len = 0
        for loss, label_num in outputs:
            total_loss += loss * label_num
            total_batch_len += label_num
        total_loss /= total_batch_len
        ppl = torch.exp(total_loss)
        
        res = {"epoch_num": self.current_epoch, "ppl": ppl, "loss": total_loss}
        
        res.update(compute_batch(label_batch, inference_result, key_word_batch))


        if not self.trainer.running_sanity_check:
            wandb.log(res)
            self.log("ppl", ppl)
            self.log("bleu-1", res['bleu-1'])

    def post_process_for_validation(self, encoded_batch, encoded_label_batch, output):
        if self.args.model_name == "BART":
            
            inference_result = [self.tokenizer.decode(sample[1:-1], skip_special_tokens=True)\
                    .replace(" ", "").replace(self.special_tokens['delimeter'], "") for sample in output]

            label_batch = [self.tokenizer.decode(sample[sample != -100], skip_special_tokens=True)\
                    .replace(" ", "").replace(self.special_tokens['delimeter'], "") for sample in encoded_label_batch]

            key_word_batch = [self.tokenizer.decode(sample, skip_special_tokens=False)\
                    .replace(" ", "").replace(self.tokenizer.cls_token, "").replace(self.tokenizer.pad_token, "")\
                    .replace(self.tokenizer.sep_token, "") for sample in encoded_batch]

        else:
            
            inference_result = [self.tokenizer.decode(output[:,1:-1], skip_special_tokens=True)\
                    .replace(" ", "").replace(self.special_tokens['delimeter'], "") for sample in output]

            label_batch = [self.tokenizer.decode(sample[sample != -100], skip_special_tokens=True)\
                    .replace(" ", "").replace(self.special_tokens['delimeter'], "") for sample in encoded_label_batch]

            delimeter_idx = [itm.index(self.delimeter_ids) for itm in encoded_batch]
            
            for delimeter_id in delimeter_idx:
                encoded_batch[:, delimeter_id:] = -100
            key_word_batch = [self.tokenizer.decode(sample, skip_special_tokens=False)\
                    .replace(" ", "").replace(self.tokenizer.cls_token, "").replace(self.tokenizer.pad_token, "")\
                    .replace(self.tokenizer.sep_token, "") for sample in encoded_batch]


        return label_batch, inference_result, key_word_batch
'''

    def test_step(self, batch, batch_idx):
        
        with open('test.json', 'a+') as f:
            for original in batch['original']:
                sentence_ids = self.tokenizer(original).input_ids
                output = self.gpt.generate(sentence_ids, max_length=1024,\
                    num_beams=10, bos_token_id = self.bos_ids, eos_token_id=self.eos_ids,\
                    pad_token_id=self.eos_ids, length_penalty=self.args.length_penalty)

                f.write(' '.join(self.tokenizer.decode(output[0]))+'\n')
            
        '''