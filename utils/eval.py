import json
import argparse
import sys
import numpy as np
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams
from rouge import Rouge

special_token = '<DELIMETER>'


def proline(line):
    return " ".join([w for w in jieba.cut("".join(line.strip().split()))])

def bleu(data):
    """
    compute rouge score
    Args:
        data (list of dict including reference and candidate):
    Returns:
            res (dict of list of scores): rouge score
    """

    res = {}
    for i in range(1, 5):
        res["bleu-%d"%i] = []

    for tmp_data in data:
        origin_candidate = tmp_data['candidate']
        origin_reference = tmp_data['reference']
        assert isinstance(origin_candidate, str)
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]

        for i in range(1, 5):
            res["bleu-%d"%i].append(sentence_bleu(references=[r.strip().split() for r in origin_reference], hypothesis=origin_candidate.strip().split(), weights=tuple([1./i for j in range(i)]))) 
            
    for i in range(1, 5):
         res["bleu-%d-cat"%i] = res["bleu-%d"%i].copy()
            
    for key in res:
        if 'cat' not in key:
            res[key] = np.mean(res[key])
        
    return res



def repetition_distinct_validation(eval_data):
    eval_data = [proline(single_eval_data) for single_eval_data in eval_data]
    result = {}
    for i in range(1, 5):
        all_ngram, all_ngram_num = {}, 0.
        for k, tmp_data in enumerate(eval_data):
            ngs = ["_".join(c) for c in ngrams(tmp_data, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)
    return result

def repetition_distinct(eval_data):
    result = {}
    for i in range(1, 5):
        all_ngram, all_ngram_num = {}, 0.
        for k, tmp_data in enumerate(eval_data):
            ngs = ["_".join(c) for c in ngrams(tmp_data["candidate"], i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)
    return result


def rouge(ipt, cand):
    rouge_name = ["rouge-1", "rouge-2", "rouge-l"]
    item_name = ["f", "p", "r"]

    res = {}
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s"%(name1, name2)] = []
    for k, (tmp_ipt, tmp_cand) in enumerate(zip(ipt, cand)):
        for tmp_ref in tmp_ipt.split(special_token):
            # print(tmp_ref.strip())
            # print(" ".join(tmp_cand))

            # tmp_ref = tmp_ref.strip()
            # tmp_hyp = " ".join(tmp_cand).strip()

            tmp_ref = " ".join([w for w in "".join(tmp_ref.strip().split())])
            tmp_hyp = " ".join([w for w in "".join(tmp_cand.strip().split())])
            # print(tmp_ref)
            # print(tmp_hyp)
            try:
                tmp_res = Rouge().get_scores(refs=tmp_ref, hyps=tmp_hyp)[0]
                for name1 in rouge_name:
                    for name2 in item_name:
                        res["%s-%s"%(name1, name2)].append(tmp_res[name1][name2])
            except:
                continue
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s-cat"%(name1, name2)] = res["%s-%s"%(name1, name2)]                
            res["%s-%s"%(name1, name2)] = np.mean(res["%s-%s"%(name1, name2)])
            
    return {"coverage": res["rouge-l-r"], "coverage-cat": res["rouge-l-r-cat"]}


def LCS(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def Recon_LCS(x, y, exclusive=True):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = LCS(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    if len(recon_list):
        return "".join(recon_list).strip()
    else:
        return ""
    # return Ngrams(recon_list, exclusive=exclusive)
    # return recon_tuple


def lcs3_dp(input_x, input_y):
    # input_y as column, input_x as row
    dp = [([0] * (len(input_y)+1)) for i in range(len(input_x)+1)]
    maxlen = maxindex = 0
    for i in range(1, len(input_x)+1):
        for j in range(1, len(input_y)+1):
            if i == 0 or j == 0:  # ?????????????????????+1
                    dp[i][j] = 0
            if input_x[i-1] == input_y[j-1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > maxlen:  # ????????????????????????????????????????????????
                    maxlen = dp[i][j]
                    maxindex = i - maxlen
                    # print('??????????????????????????????:%s' % maxlen)
                    # print('?????????????????????:%s' % input_x[maxindex:maxindex + maxlen])
            else:
                dp[i][j] = 0
    # for dp_line in dp:
    #     print(dp_line)
    return input_x[maxindex:maxindex + maxlen]

def inversenum(a):
    num = 0
    all_num = 0
    for i in range(0,len(a)):
        for j in range(i,len(a)):
            if a[i] > a[j]:
                num += 1
            all_num += 1
    return num / float(all_num)

def find_all(sub,s):
	index_list = []
	index = s.find(sub)
	while index != -1:
		index_list.append(index)
		index = s.find(sub,index+1)
	
	if len(index_list) > 0:
		return index_list
	else:
		return -1

def order(ipt, cand, kw2id):
    num = []
    for k, (tmp_ipt, tmp_cand, tmp_kw2id) in enumerate(zip(ipt, cand, kw2id)):
        # all_pos = [[]]
        pos = []
        kw_list = list(tmp_kw2id.keys())
        kw_list.reverse()

        for tmp_ref in kw_list:
            tmp_ref = "".join(tmp_ref.strip().split())
            tmp_hyp = "".join(tmp_cand.strip().split())
            lcs = lcs3_dp(tmp_ref, tmp_hyp)
            if len(lcs)>1:
                pos.append(tmp_hyp.find(lcs))
            else:
                pos.append(-1)
        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])

        new_rank = [-1 for _ in idlist]
        for idl, ord in zip(idlist, orderlist):
            new_rank[idl] = tmp_kw2id[kw_list[ord]]
        num.append(1-inversenum(new_rank))

    return {"order": np.mean(num), "order-cat": num}



def load_file(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))
        f.close()
    return data


def overall_compare(res):
    small = [26.58, 16.04, 17.90, 31.38, 83.64, 63.15]
    true = [100, 100, 23.47, 42.17, 100, 100]
    predict = [res["bleu-1"], res["bleu-2"], res["distinct-3"], res["distinct-4"], res["coverage"], res["order"]]
    sum = 0
    wi = []
    for a, b in zip(true, small):
        wi.append(a / b)
        sum += a / b
    overall = 0
    for a, w in zip(predict, wi):
        overall += a * (w / sum)
    return overall

def compute_batch(golden_batch, pred_batch, key_word_batch, return_dict=True):

    #ipt = ["#".join(g["outline"]) for g in golden_data]
    ipt = key_word_batch
    truth = golden_batch
    pred = pred_batch
    kw2id = []
    for i1, t1 in zip(ipt, truth):
        kw_list = i1.strip().split(special_token)
        pos = [t1.strip().find(kw.strip()) for kw in kw_list]

        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])
        kw2id.append({})
        for idl, ord in zip(idlist, orderlist):
            kw2id[-1][kw_list[ord]] = idl


    eval_data = [{"reference": proline(g), "candidate": proline(p)} for g, p in zip(truth, pred)]
    res = bleu(eval_data)
    #res.update(repetition_distinct(eval_data))
    res.update(rouge(ipt=ipt, cand=pred))
    res.update(order(ipt=ipt, cand=pred, kw2id=kw2id))
    #res.update({"overall": overall_compare(res)})
    # for key in res:
    #     res[key] = "_"
    return res

def compute(golden_file, pred_file, return_dict=True):
    golden_data = load_file(golden_file)
    #pred_data = load_file(pred_file)

    pred_data = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            pred_data.append({"story": line})
        f.close()

    if len(golden_data) != len(pred_data):
        raise RuntimeError("Wrong Predictions")

    ipt = [special_token.join(g["outline"]) for g in golden_data]
    truth = [g["story"] for g in golden_data]
    pred = [p["story"] for p in pred_data]

    kw2id = []
    for i1, t1 in zip(ipt, truth):
        kw_list = i1.strip().split(special_token)
        pos = [t1.strip().find(kw.strip()) for kw in kw_list]

        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])
        kw2id.append({})
        for idl, ord in zip(idlist, orderlist):
            kw2id[-1][kw_list[ord]] = idl


    eval_data = [{"reference": proline(g["story"]), "candidate": proline(p["story"])} for g, p in zip(golden_data, pred_data)]
    res = bleu(eval_data)
    res.update(repetition_distinct(eval_data))
    res.update(rouge(ipt=ipt, cand=pred))
    res.update(order(ipt=ipt, cand=pred, kw2id=kw2id))
    res.update({"overall": overall_compare(res)})
    
    res_1 = {"bleu-1": res['bleu-1'], "bleu-2": res['bleu-2'], "distinct-3": res['distinct-3'], "distinct-4": res['distinct-4'], \
            'order': res['order'], "coverage": res['coverage'], 'overall': res['overall']}
    # for key in res:
    #     res[key] = "_"
    return res_1

def main():
    argv = sys.argv
    print("???????????????{}, ?????????: {}".format(argv[1], argv[2]))
    print(compute(argv[2], argv[1]))


if __name__ == '__main__':
    main()
