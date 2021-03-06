from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
import torch

class OutGenDataset(Dataset):
    def __init__(self, root, task_name, special_tokens, model_name="BART"):
        self.root = root
        self.task_name = task_name
        self.special_tokens = special_tokens
        self.delimeter = self.special_tokens['delimeter']
        self.sep = self.special_tokens['sep']
        self.eos = self.special_tokens['eos']
        self.bos = self.special_tokens['bos']
        self.model_name = model_name
        self.preprocess(task_name)

    def data_concat(self, json_data):
        if self.model_name == 'CPM':
            outline = self.delimeter.join(json_data['outline']) 
            source = f"{self.bos}{outline}{self.sep}"
            target = f"{self.bos}{outline}{self.sep}{json_data['story']}{self.eos}"
        else:
            outline = self.delimeter.join(json_data['outline']) 
            source = f"<WORD>{outline}{self.eos}"
            target = f"{self.bos}{json_data['story']}{self.eos}"

        return {'source': source, 'target': target}

    def preprocess(self, task_name):
        #print(os.path.join(self.root, f'{task_name}.jsonl'))
        assert os.path.isfile(os.path.join(self.root, f'{task_name}.jsonl')) 

        if not os.path.isfile(os.path.join(self.root, f'{self.model_name}_{task_name}.sample')):
            samples = []
            with open(os.path.join(self.root, f'{task_name}.jsonl'), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    json_data = json.loads(line)
                    samples.append(self.data_concat(json_data))

            with open(os.path.join(self.root, f'{self.model_name}_{task_name}.sample'), 'w', encoding='utf-8') as f_source:
                for line in samples:
                    json.dump(line, f_source, ensure_ascii=False)
                    f_source.write("\n")
        else:
            samples = []

            with open(os.path.join(self.root, f'{self.model_name}_{task_name}.sample'), 'r', encoding='utf-8') as f_target:
                for line in f_target.readlines():
                    samples.append(json.loads(line))
                
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        
        return self.samples[i]

class OutGenDataModule(LightningDataModule):
    def __init__(self, root, local_tokenizer, args, special_tokens) -> None:
        super().__init__()
        self.root = root
        self.delimeter = special_tokens['delimeter']
        self.args = args
        self.tokenizer = local_tokenizer
        self.special_tokens = special_tokens
        self.delimeter_ids = self.tokenizer.convert_tokens_to_ids(self.delimeter)
        self.pad_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.eos_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.cls_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)

    def collate_fn_cpm(self, batch):
        label_batch = [ans['target'] for ans in batch]

        encoded_batch = self.tokenizer(label_batch, add_special_tokens=False, padding=True,\
             max_length=self.args.max_length, truncation=True)

        inputs = torch.tensor(encoded_batch.input_ids)
        
        
        label_batch = [ans['target'] for ans in batch]

        encoded_label_batch = torch.tensor(self.tokenizer(label_batch, padding=True, add_special_tokens=False,\
             max_length=self.args.max_length, truncation=True).input_ids)

        encoded_batch_list = encoded_label_batch.tolist()
        delimeter_idx = [itm.index(self.sep_ids) for itm in encoded_batch_list]

        for labels in encoded_label_batch:
            if labels[-1] != self.eos_ids:
                labels[-1] = self.eos_ids

        for idx, delimeter_id in enumerate(delimeter_idx):
            encoded_label_batch[idx, :delimeter_id + 1] = self.pad_ids
            

        generate_batch = [ans['source'] for ans in batch]

        encoded_generate_batch = torch.tensor(self.tokenizer(generate_batch, padding=True, add_special_tokens=False,\
             max_length=self.args.max_length, truncation=True).input_ids)

        for labels in encoded_generate_batch:
            if labels[-1] != self.delimeter_ids:
                labels[-1] = self.delimeter_ids

        delimeter_idx = [itm.index(self.delimeter_ids) for itm in encoded_batch_list]
        for idx, delimeter_id in enumerate(delimeter_idx):
            encoded_label_batch[idx, delimeter_id+1:] = self.pad_ids


        encoded_label_batch[encoded_label_batch == self.pad_ids] = -100
        return inputs, torch.tensor(encoded_batch.attention_mask), encoded_label_batch, encoded_generate_batch
        
    def collate_fn(self, batch):
        context_batch = [ans['source'] for ans in batch]
        # use the predefined model length 
        encoded_batch = self.tokenizer(context_batch, padding=True, max_length=self.args.max_length, truncation=True)

        label_batch = [ans['target'] for ans in batch]

        encoded_label_batch = torch.tensor(self.tokenizer(label_batch, padding=True, add_special_tokens=False, max_length=self.args.max_length, \
            truncation= True).input_ids)
        for labels in encoded_label_batch:
            if labels[-1] != self.pad_ids:
                labels[-1] = self.eos_ids
        encoded_label_batch[encoded_label_batch == self.pad_ids] = -100
        #encoded_label_batch = torch.tensor(encoded_batch.input_ids.copy())
        #encoded_label_batch[encoded_label_batch == self.pad_ids] = -100

        answer_batch = [ans['target'].replace(self.tokenizer.eos_token, "") for ans in batch]

        return torch.tensor(encoded_batch.input_ids), torch.tensor(encoded_batch.attention_mask), encoded_label_batch, answer_batch

    def setup(self, stage=None):
        self.train_dataset = OutGenDataset(self.root, 'train', special_tokens=self.special_tokens, model_name=self.args.model_name)
        self.val_dataset = OutGenDataset(self.root, 'val', special_tokens=self.special_tokens, model_name=self.args.model_name)
        self.test_dataset = OutGenDataset(self.root, 'test', special_tokens=self.special_tokens, model_name=self.args.model_name)

    def train_dataloader(self):
        #process dataset frist 
        if self.args.model_name == 'CPM':
            loader = DataLoader(
                self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True, collate_fn= self.collate_fn_cpm
            )
        else:
            loader = DataLoader(
                self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True, collate_fn= self.collate_fn
            )
        return loader

    def val_dataloader(self):
        if self.args.model_name == 'CPM':
            loader = DataLoader(
                self.val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.workers, pin_memory=True, collate_fn= self.collate_fn_cpm
            )
        else:
            loader = DataLoader(
                self.val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.workers, pin_memory=True, collate_fn= self.collate_fn
            )
        return loader
    
    def test_dataloader(self):
        if self.args.model_name == 'CPM':
            loader = DataLoader(
                self.test_dataset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.workers, pin_memory=True, collate_fn= self.collate_fn_cpm
            )
        else:
            loader = DataLoader(
                self.test_dataset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.workers, pin_memory=True, collate_fn= self.collate_fn
            )
        #torch.save(self.test_rocdataset, os.path.join(self.args.save_dataset_dir, self.jobs + '_test_dataset_1.pth'))
        return loader
