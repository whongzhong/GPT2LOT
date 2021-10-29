import torch
from torchmetrics import Metric
from utils.eval import repetition_distinct_validation
    

class Distinct(Metric):
    def __init__(self, tokenizer, special_tokens, dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("generation_result", default=[], dist_reduce_fx="cat")
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens

    def update(self, generation_result):
        self.generation_result.append(generation_result)

    def compute(self):
        inference_result = [self.tokenizer.decode(sample[1:-1], skip_special_tokens=True)\
                    .replace(" ", "").replace(self.special_tokens['delimeter'], "") for sample in self.generation_result]
        res = repetition_distinct_validation(inference_result)
        #import ipdb; ipdb.set_trace()
        return res