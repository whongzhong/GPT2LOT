import torch
from torchmetrics import Metric
from utils.eval import repetition_distinct_validation, overall_compare
    

class Distinct(Metric):
    def __init__(self, tokenizer, special_tokens, dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("generation_result", default=[], dist_reduce_fx="cat")
        
        self.add_state("bleu_1", default=[], dist_reduce_fx="cat")
        self.add_state("bleu_2", default=[], dist_reduce_fx="cat")
        
        self.add_state("order", default=[], dist_reduce_fx="cat")
        self.add_state("coverage", default=[], dist_reduce_fx="cat")
        
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

    def update(self, generation_result, bleu_1, bleu_2, coverage, order):
        self.generation_result.append(generation_result)
        
        
        self.bleu_1.append(bleu_1)
        self.bleu_2.append(bleu_2)
        self.coverage.append(coverage)
        self.order.append(order)

    def compute(self):
        inference_result = [self.tokenizer.decode(sample[1:-1], skip_special_tokens=True)\
                    .replace(" ", "").replace(self.special_tokens['delimeter'], "") for sample in self.generation_result]
        
        print("distinctlen: %d"%len(inference_result))
        res = repetition_distinct_validation(inference_result)
        
        #print(f"bleu-1-len{self.bleu_1.shape}")
        #print(f"bleu-2-len{self.bleu_2.shape}")
        #print(f"coverae-len{self.coverage.shape}")
        #print(f"order-len{self.order.shape}")
        
        res['bleu-1'] = torch.mean(self.bleu_1)
        res['bleu-2'] = torch.mean(self.bleu_2)
        
        res['coverage'] = torch.mean(self.coverage)
        res['order'] = torch.mean(self.order)
        
        res.update({"overall": overall_compare(res)})
        # overall compute?
        #import ipdb; ipdb.set_trace()
        return res