import json
import os

def gen_ordered_datasets(datapath, filename, out_put_filename, replace=True):
    datasets = []
    with open(os.path.join(datapath, filename), 'r') as f:
        for line in f.readlines():
            sample = json.loads(line)
            outline = sample['outline']
            order = {}
            for outline_sample, order_sample in zip(outline, sample['order']):
                order[outline_sample] = order_sample
            ordered_outline = sorted(outline, key=lambda x:order[x])
            if replace:
                sample['outline'] = ordered_outline
            else:
                sample['ordered_outline'] = ordered_outline
            datasets.append(sample)
    with open(os.path.join(datapath, out_put_filename), 'w') as f:
        for sample in datasets:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    gen_ordered_datasets('./LOTdatasets/orderd', 'train_order.jsonl', 'train.jsonl')
    gen_ordered_datasets('./LOTdatasets/orderd', 'valid_order.jsonl', 'val.jsonl')
    gen_ordered_datasets('./LOTdatasets/orderd', 'valid_order.jsonl', 'test.jsonl')
    gen_ordered_datasets('./LOTdatasets/orderd', 'extral_order.jsonl', 'extra_ordered.jsonl')