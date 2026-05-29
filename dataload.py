import ast
from torch.utils.data.dataset import Dataset
import pandas as pd
import torch  
from io import StringIO  

    
class GET_DATA_PPL(Dataset):
    def __init__(self, file_path):
        # self.data = pd.read_csv(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        df = pd.read_csv(StringIO(text))
        self.data = df
        
        self.getsample()
        
    def __len__(self):
        return len(self.data)


    def _safe_parse_list(self, s):
        import re, json
        if isinstance(s, list):
            return s
        if not isinstance(s, str):
            return []
        t = s.strip()
        if t.startswith('[') and t.endswith(']'):
            try:
                return json.loads(t)
            except Exception:
                try:
                    return ast.literal_eval(t)
                except Exception:
                    pass
        m = re.match(r"^(array|tensor|list)\s*\((.*)\)\s*$", t)
        if m:
            inner = m.group(2).strip()
            br = re.search(r"\[(.*)\]", inner)
            if br:
                content = '[' + br.group(1) + ']'
                try:
                    return json.loads(content)
                except Exception:
                    try:
                        return ast.literal_eval(content)
                    except Exception:
                        pass
        nums = re.findall(r"-?\d+\.?\d*", t)
        if nums:
            if all('.' not in x for x in nums):
                return [int(x) for x in nums]
            else:
                return [float(x) for x in nums]
        return []

    def getsample(self):
        self.sample = []
        for idx in range(len(self.data)):
            ll_tokens_1_and_2_and_3 = self._safe_parse_list(self.data.iloc[idx, 17])
            ll_tokens_3 = self._safe_parse_list(self.data.iloc[idx, 11])
            diff_3= self.get_difference(ll_tokens_1_and_2_and_3, ll_tokens_3)
            diff_3 = self._pad_tokens(diff_3, 512)

                
            self.sample.append({
                'article_id': self.data.iloc[idx, 0],
                'input_sentences_list': self._safe_parse_list(self.data.iloc[idx, 2]),
                'input_sentences': self.data.iloc[idx, 3],
                'label': self.data.iloc[idx, 4],
                'label_np':  self._parse_label_np(self.data.iloc[idx, 5]),
                'diff_3': torch.tensor(diff_3),
            })
            

    def __getitem__(self, i):
        sentence_sample = self.sample[i]
        return sentence_sample


    def _parse_label_np(self, label_np_str):
        label_np_str = label_np_str.strip('[]')  
        label_np_list = [int(x) for x in label_np_str.split()]  
        return label_np_list
    
    def _pad_tokens(self, tokens_list,length):
        if len(tokens_list) < length:
            tokens_list.extend([0] * (length - len(tokens_list)))
        elif len(tokens_list) > length:
            tokens_list = tokens_list[:length]
        return tokens_list
    
    def get_difference(self, tokens_list_1, tokens_list_2):
        if len(tokens_list_1) < len(tokens_list_2):
            raise ValueError("error: tokens_list_1 should be longer than or equal to tokens_list_2")
        A_tail = tokens_list_1[-len(tokens_list_2):]
        difference = [abs(a - b) for a, b in zip(A_tail, tokens_list_2)]
        return difference
    
    
    
