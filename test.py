import transformers
import torch
from tqdm import tqdm
import argparse
import numpy as np
import re
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import string
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import os
from os.path import join, dirname, basename
from pathlib import Path
from BBPE_model import BBPEmodel

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,help='path to goodnews file')
parser.add_argument('--sentence_head_folder', type=str)
parser.add_argument('--cache_dir', type=str)

parser.add_argument('--window_size', type=int)
parser.add_argument('--window_step', type=int)
parser.add_argument('--article_num', type=int) # 2 for debug
parser.add_argument('--save_folder', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--best_model', type=str)
parser.add_argument('--explain', type=str)

args = parser.parse_args()
print(args)





DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBERTA_MAX_TEXT_LENGTH=512
GPT2_MAX_TEXT_LENGTH=1024
    
    
def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[。！？.!?])\s+", text)
    rows = [c.strip() for c in chunks if c.strip()]
    if rows:
        return rows
    return [line.strip() for line in text.splitlines() if line.strip()] or [text]


def is_only_punctuation_or_digit_or_single_letter(sentence: str) -> bool:
    sentence = sentence.replace(" ", "")
    if not sentence:
        return True
    if all(char in string.punctuation for char in sentence):
        return True
    if sentence.isdigit():
        return True
    if len(sentence) == 1 and sentence.isalpha():
        return True
    if len(sentence.split()) == 1:
        return True
    return False


def pad_tokens(tokens_list: list[float], length: int = 512) -> list[float]:
    if len(tokens_list) < length:
        tokens_list = tokens_list + ([0] * (length - len(tokens_list)))
    elif len(tokens_list) > length:
        tokens_list = tokens_list[:length]
    return tokens_list


def get_difference(tokens_list_1: list[float], tokens_list_2: list[float]) -> list[float]:
    if len(tokens_list_1) < len(tokens_list_2):
        return [0.0 for _ in tokens_list_2]
    tail = tokens_list_1[-len(tokens_list_2):]
    return [abs(a - b) for a, b in zip(tail, tokens_list_2)]


class SentencePredictor:
    def __init__(self, sentence_head_folder: str, best_model: str, window_size: int, window_step: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.window_step = window_step
        self.model_ppl = BBPEmodel()
        model_path = Path(sentence_head_folder) / best_model
        try:
            self.sentence_head_model = torch.load(
                str(model_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            self.sentence_head_model = torch.load(str(model_path), map_location=self.device)
        
        
        self.sentence_head_model.eval()
        self.tokenizer = self.sentence_head_model.deberta_tokenizer

    def get_PPL(text_data):
        sen1=text_data[0]
        if len(text_data)==1:
            sen2=text_data[0]
            sen3=text_data[0]
        elif len(text_data)==2:
            sen2=text_data[1]
            sen3=text_data[1]
        else:
            sen2=text_data[1]
            sen3=text_data[2]
        
        if is_only_punctuation_or_digit_or_single_letter(sen1):
            sen1 = sen1 +' ' + sen1  
        if is_only_punctuation_or_digit_or_single_letter(sen2):
            sen2 = sen2+ ' ' + sen2
        if is_only_punctuation_or_digit_or_single_letter(sen3):
            sen3 = sen3+ ' ' + sen3

        sen1_and_sen2_and_sen3 = sen1 +' '+ sen2 +' '+ sen3
        _,_,ll_token3=BBPEmodel.ppl(text=sen3)
        _,_,ll_token1_and_2_and_3=BBPEmodel.ppl(text=sen1_and_sen2_and_sen3)
        diff3_123=get_difference(ll_token1_and_2_and_3,ll_token3)

        diff3_123=torch.tensor(pad_tokens(diff3_123))

        # return diff_2,diff_3
        return diff3_123

    def get_supervised_model_prediction(self, model, tokenizer, sentence_list, DEVICE, pos_bit=0, window_size=args.window_size, window_step=args.window_step):
        with torch.no_grad():
            preds = []

            each_sample_preds = []
            majority_vote_preds = [[] for i in range(len(sentence_list))]
            # print("sentence_list", len(sentence_list))

            for window_start in range(0, max(1, len(sentence_list)-window_size+1), window_step):
                text_data = sentence_list[window_start : window_start+window_size]
                text_merge = " ".join(text_data)
                
                diff_3_123 = self.get_PPL(text_data)
            
                sentence_feature = model.extract_deberta_PPL(text=text_merge,diff_3=diff_3_123,batchsize=1)
                prediction_score = torch.sigmoid(model(sentence_feature)).tolist()[0]
                # print("prediction_score", prediction_score)

                each_sample_preds.append(prediction_score[1])
                try:
                    idx=0
                    for vote_idx in range(window_start, window_start+window_size):
                        majority_vote_preds[vote_idx].append(prediction_score[idx])
                        idx+=1
                except:
                    idx=0
                    for vote_idx in range(window_start, min(window_start+window_size, len(sentence_list))):
                        majority_vote_preds[vote_idx].append(prediction_score[idx])
                        idx+=1

        # print("majority_vote_preds", majority_vote_preds)
        # majority_vote_preds_mean = [sum(sub_list)/len(sub_list) for sub_list in majority_vote_preds]
        # # majority_vote_preds_mean = [(sum(sub_list)+max(sub_list))/(len(sub_list)+1) for sub_list in majority_vote_preds]
        
        majority_vote_preds_mean = []  
        for sub_list in majority_vote_preds:
            if len(sub_list)<=2:
                mean_value = sum(sub_list)/len(sub_list)
            else:
                confidence_weights = [abs(p - 0.5) * 2 for p in sub_list]
                total_weight = sum(confidence_weights)
                if total_weight == 0:  
                    normalized_weights = [1/len(confidence_weights)] * len(confidence_weights)
                else:
                    normalized_weights = [w/total_weight for w in confidence_weights]
                mean_value = sum(p * w for p, w in zip(sub_list, normalized_weights))
                # mean_value = 0.3*sub_list[0]+0.4*sub_list[1]+0.3*sub_list[2]
            majority_vote_preds_mean.append(mean_value)

        each_sample_preds = [majority_vote_preds_mean[0]] + each_sample_preds + [majority_vote_preds_mean[-1]]
        return majority_vote_preds_mean, majority_vote_preds, each_sample_preds,majority_vote_preds
    
    def run_supervised_experiment_sentence_head(self, data, cache_dir, DEVICE, pos_bit=0, window_size=args.window_size, sentence_head_model=None):
        tokenizer = sentence_head_model.deberta_tokenizer
        test_preds_list = []
        test_gt_list = []
        AP_list = []
        acc_list = []
        dataset_single_preds_list = []
        dataset_preds_list = []
        dataset_gt_list = [] # calculate average precision all together
        invalid_num=0
        preds=[]

        for sample in tqdm(data, desc="run evaluation on articles"):
            try:  # For GoodNews, VisualNews, WikiText datasets
                article_id = sample['article_id']
                # print(article_id)
                test_mixed_text = " ".join(sample['merge_sentences'])
                test_mixed_sentences = sample['merge_sentences']
                label = sample['config_dict']['mixed_labels']  # sentence label
                num_chunks = sample['config_dict']['number_of_chunks']
                model_name = sample['config_dict']['model_name']
            except:  # For GhostBuster datasets
                test_mixed_sentences = sample['return_sentences']
                label = sample['return_labels']


            test_preds,_,test_preds_single,m_preds = self.get_supervised_model_prediction(
                sentence_head_model, tokenizer, test_mixed_sentences, DEVICE=DEVICE, pos_bit=pos_bit, window_size=window_size)# window_size: how many sentences within a window
            AP = average_precision_score(y_true=np.array(label), y_score=np.array(test_preds))
            acc = accuracy_score(y_true=np.array(label), y_pred=np.array(test_preds).round())
            # AP_single = average_precision_score(y_true=np.array(label), y_score=np.array(test_preds))

            assert len(test_preds)==len(label), "check label for each article"
            test_preds_list.append(np.array(test_preds))
            test_gt_list.append(np.array(label))
            AP_list.append(AP)
            acc_list.append(acc)

            dataset_preds_list += test_preds
            dataset_gt_list += label
            dataset_single_preds_list += test_preds_single
            preds += m_preds

if __name__=="__main__":
    

    if args.data_path.split(".")[-1]=="jsonl":
        with open(args.data_path, 'r') as f:
            data = []
            for l_no, l in enumerate(f):
                data.append(json.loads(l))
    elif args.data_path.split(".")[-1]=='json':
        with open(args.data_path,'r') as f:
            data = json.load(f)


    sentence_head_model = torch.load(os.path.join(args.sentence_head_folder, args.best_model), map_location=DEVICE)
    results = SentencePredictor.run_supervised_experiment_sentence_head(data=data[0: args.article_num], cache_dir=args.cache_dir,
                              DEVICE=DEVICE, pos_bit=0, window_size=args.window_size, sentence_head_model=sentence_head_model)
    print("AP results: ", results['AP'])
    print("dataset_AP results: ", results['dataset_AP'])
    print("dataset_acc results: ", results['dataset_acc'])

    # print(results['AP_list'])


    AP_results = {
        "explain":args.explain,
        "AP": results['AP'],
        "dataset_AP": results['dataset_AP'],
        "dataset_acc": results['dataset_acc'],
        # "dataset_singleAP": results['dataset_singleAP'],
        "args": vars(args),
        "AP_list": results['AP_list'],
        "ground_truth": results["ground_truth"],
        "prediction": results["prediction"],
        "preds": results["preds"]
    }

    os.makedirs(args.save_folder,exist_ok=True)
    with open(join(args.save_folder, args.save_name+".json"),"w") as f:
        json.dump(AP_results,f,indent=2, default=str)






















