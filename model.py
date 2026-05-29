import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
import string

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self,hidden_size=1536,
                 num_labels=4,  
                 dropout=0.1,
                 deberta_detector_name="deberta-v3-large",
                 cache_dir: str = "deberta-v3-large",
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.out2= nn.Linear(512, num_labels)

        if deberta_detector_name:
            print("cache_dir:", cache_dir)
            self.deberta_tokenizer = transformers.AutoTokenizer.from_pretrained(deberta_detector_name, cache_dir=cache_dir)
            self.deberta_detector = transformers.AutoModelForSequenceClassification.from_pretrained(
                deberta_detector_name, cache_dir=cache_dir).to(DEVICE)
            self.deberta_detector.eval()
            
    def forward(self, features):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features).to(DEVICE)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x # (batch_size, num_labels)
    
    def extract_deberta_feature(self, text):
        sample_manipulated_article_token = self.deberta_tokenizer(text,
                                                             padding='max_length',
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors="pt").to(
            DEVICE)  # (1, text_length), text_length should be smaller than 512
        sample_manipulated_article_embeddings = self.deberta_detector(**sample_manipulated_article_token,
                                                                 output_hidden_states=True, return_dict=True)
        last_hidden_state = sample_manipulated_article_embeddings['hidden_states'][-1] 
        return last_hidden_state
    
    def extract_deberta_PPL(self,text,diff_3,batchsize):
        sample_manipulated_article_token = self.deberta_tokenizer(text,
                                                             padding='max_length',  # longest, max_length, False
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors="pt").to( DEVICE)  # (1, text_length), text_length should be smaller than 512
        sample_manipulated_article_embeddings = self.deberta_detector(**sample_manipulated_article_token,
                                                                 output_hidden_states=True, return_dict=True)
        last_hidden_state = sample_manipulated_article_embeddings['hidden_states'][-1] 
        cls=last_hidden_state[:,0,:]
        cls_temp=torch.zeros(batchsize, 1536)
        cls_temp[:, :1024] = cls
        cls_temp[:, 1024:1536] = diff_3
        return cls_temp 
    
    def extract_deberta_PPL_23(self,text,diff_2,diff_3,batchsize):
        sample_manipulated_article_token = self.deberta_tokenizer(text,
                                                             padding='max_length',  # longest, max_length, False
                                                             truncation=True,
                                                             max_length=512,
                                                             return_tensors="pt").to( DEVICE)  # (1, text_length), text_length should be smaller than 512
        sample_manipulated_article_embeddings = self.deberta_detector(**sample_manipulated_article_token,
                                                                 output_hidden_states=True, return_dict=True)
        last_hidden_state = sample_manipulated_article_embeddings['hidden_states'][-1] 
        cls=last_hidden_state[:,0,:]
        cls_temp=torch.zeros(batchsize, 1536)
        cls_temp[:, :1024] = cls
        cls_temp[:, 1024:1280] = diff_2
        cls_temp[:, 1280:] = diff_3
        return cls_temp 

