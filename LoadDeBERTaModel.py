
import os, re, gc
from ast import literal_eval
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizerFast, RobertaModel

#### BASE_URL = "../input/nbme-score-clinical-patient-notes"
BASE_URL = "./data"


###Hyperparameters
hyperparameters = {
    "max_length": 512,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    #### "model_name": "../input/deberta-v3-large/deberta-v3-large",
    ##"model_name": "./model/huggingface-bert/microsoft/deberta-v3-large",  
    "model_name": "./model/huggingface-bert/microsoft/deberta-v2-xlarge",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 1268,
    "batch_size": 1,
    "RTDM": True, ### replaced token detection, RTD model
}

def process_feature_text(text):
    return text.replace("-OR-", ";-").replace("-", " ")


from DeBERTa_RTD.DeBERTa.apps.models.replaced_token_detection_model import ReplacedTokenDetectionModel
RTD_model_path = "/home/xyb/Project/Kaggle/NBME/model/huggingface-bert/microsoft/deberta-v2-xlarge/pytorch_model.bin"
RTD_model_cfg = "/home/xyb/Project/Kaggle/NBME/DeBERTa_RTD/experiments/language_model/deberta_xlarge.json"

"""
Model
Lets use BERT base Architecture
Also Used 3 FC layers
Comments: 3 layers improve accuracy 2% on public score
"""
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.modelCfg = AutoConfig.from_pretrained(config['model_name'], output_hidden_states=True)
        if self.config['RTDM']:
            self.fc = nn.Linear(512, 1)
            self.model = ReplacedTokenDetectionModel.load_model(model_path=RTD_model_path, model_config=RTD_model_cfg)
        else:
            self.fc = nn.Linear(self.modelCfg.hidden_size, 1)
            self.model = AutoModel.from_pretrained(config['model_name'], config=self.modelCfg)  # DeBERTa Model
            self._init_weights(self.fc)
        
        if 'deberta-v2-xxlarge' in config['model_name']:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in config['model_name']:
            ##self.model.embeddings.requires_grad_(False)
            ##self.model.encoder.layer[:12].requires_grad_(False) # 冻结12/24
            pass

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.modelCfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.modelCfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        if self.config['RTDM']:
            outputs = self.model(input_ids=input_ids, input_mask=attention_mask)
            logits = outputs['logits']
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.fc(self.dropout(outputs[0])).squeeze(-1)        
        return logits

def create_test_df():
    feats = pd.read_csv(f"{BASE_URL}/features.csv")
    feats.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    
    notes = pd.read_csv(f"{BASE_URL}/patient_notes.csv")
    test = pd.read_csv(f"{BASE_URL}/test.csv")

    merged = test.merge(notes, how = "left")
    merged = merged.merge(feats, how = "left")

    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ")
    
    def clean_spaces(text):
        text = re.sub('\n', ' ', text)
        text = re.sub('\t', ' ', text)
        text = re.sub('\r', ' ', text)
        return text
    
    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    merged["feature_text"] = merged["feature_text"].apply(clean_spaces)
    ### merged["feature_text"] = merged["feature_text"].apply(lambda x: x.lower())

    merged["pn_history"] = merged["pn_history"].apply(lambda x: x.strip())
    merged["pn_history"] = merged["pn_history"].apply(clean_spaces)
    ### merged["pn_history"] = merged["pn_history"].apply(lambda x: x.lower())

    return merged


def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        pred = 1 / (1 + np.exp(-pred))
        start_idx = None
        end_idx = None
        current_preds = []
        for pred, offset, seq_id in zip(pred, offsets, seq_ids):
            if seq_id is None or seq_id == 0:
                continue

            if pred > 0.5:
                if start_idx is None:
                    if offset[0] == 0:
                        start_idx = offset[0]
                    else:
                        start_idx = offset[0] + 1
                end_idx = offset[1]
            elif start_idx is not None:
                if test:
                    current_preds.append(f"{start_idx} {end_idx}")
                else:
                    current_preds.append((start_idx, end_idx))
                start_idx = None
        if test:
            all_predictions.append(";".join(current_preds))
        else:
            all_predictions.append(current_preds)
            
    return all_predictions

def get_location_predictions_deberta(char_preds, texts):
    all_predictions = []
    for pred, text in zip(char_preds, texts):
        current_preds = []
        
        start_idx = None
        for idx, p in enumerate(pred):
            if p > 0.5:
                # DeBERTa can start predictions with a space, but this should never match the ground truth 
                if start_idx is None and not text[idx].isspace():
                    start_idx = idx
                end_idx = idx+1
            elif start_idx is not None:
                current_preds.append(f"{start_idx} {end_idx}")
                start_idx = None
                
        if start_idx is not None:
            current_preds.append(f"{start_idx} {end_idx}")
        
        all_predictions.append("; ".join(current_preds))
    
    return all_predictions

class SubmissionDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data.loc[idx]
        tokenized = self.tokenizer(
            example["feature_text"],
            example["pn_history"],
            truncation = self.config['truncation'],
            max_length = self.config['max_length'],
            padding = self.config['padding'],
            return_offsets_mapping = self.config['return_offsets_mapping']
        )
        tokenized["sequence_ids"] = tokenized.sequence_ids()

        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        ### token_type_ids = np.array(tokenized["token_type_ids"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")

        ### return input_ids, attention_mask, token_type_ids, offset_mapping, sequence_ids
        return input_ids, attention_mask, offset_mapping, sequence_ids



def add_to_char_preds(char_preds, model_predictions, offsets_list, seq_ids_list): 
    for idx, (preds, offsets, seq_ids) in enumerate(zip(model_predictions, offsets_list, seq_ids_list)):
        #preds = sigmoid(preds)
        preds = 1 / (1 + np.exp(-preds))
        prev_offset = None
        for p, o, s_id in zip(preds, offsets, seq_ids):
            if s_id is None or s_id == 0:
                continue
            shift=0
            if prev_offset is not None:
                if prev_offset[1] < o[0]:
                    shift = 1
            char_preds[idx][o[0]-shift:o[1]] += p
            prev_offset = o
            
    return char_preds


test_df = create_test_df()

DEVICE = "cuda"

from transformers.models.deberta_v2 import DebertaV2TokenizerFast
tokenizer = DebertaV2TokenizerFast.from_pretrained(hyperparameters['model_name'])

submission_data = SubmissionDataset(test_df, tokenizer, hyperparameters)
submission_dataloader = DataLoader(submission_data, batch_size=hyperparameters['batch_size'], shuffle=False)

#### RoBERTa model with 5 folds
N_FOLDS = 1

all_preds = None
offsets = []
seq_ids = []

char_preds = [np.zeros(len(t)) for t in test_df["pn_history"]]

for fold in range(N_FOLDS):
    model = CustomModel(hyperparameters).to(DEVICE)
    model.load_state_dict(torch.load(f'/home/xyb/Project/Kaggle/NBME/upload/nbme_DeBerta-V2-XLarge_fold-0-BV.pth', map_location = DEVICE))
    model.eval()
    preds = []
    
    for batch in tqdm(submission_dataloader):
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        

        ### logits = model(input_ids, attention_mask, token_type_ids)
        logits = model(input_ids, attention_mask)

        preds.append(logits.detach().cpu().numpy())
        if fold == 0:
            ### token_type_ids = batch[2].to(DEVICE)
            offset_mapping = batch[2]
            sequence_ids = batch[3]
            offsets.append(offset_mapping.numpy())
            seq_ids.append(sequence_ids.numpy())
        
        #del input_ids, attention_mask, logits
            
    preds = np.concatenate(preds, axis=0)
    if all_preds is None:
        all_preds = np.array(preds).astype(np.float32)
    else:
        all_preds += np.array(preds).astype(np.float32)

    #gc.collect()
    #torch.cuda.empty_cache()

    char_preds = add_to_char_preds(char_preds, all_preds, np.concatenate(offsets, axis=0), np.concatenate(seq_ids, axis=0))

char_preds = [cp/N_FOLDS for cp in char_preds]
location_preds = get_location_predictions_deberta(char_preds, test_df["pn_history"])

"""
all_preds /= N_FOLDS
all_preds = all_preds.squeeze()

offsets = np.concatenate(offsets, axis=0)
seq_ids = np.concatenate(seq_ids, axis=0)

print(all_preds.shape, offsets.shape, seq_ids.shape)
"""

## submission
#location_preds = get_location_predictions(all_preds, offsets, seq_ids, test=True)

test_df["location"] = location_preds
test_df[["id", "location"]].to_csv("submission.csv", index = False)
pd.read_csv("submission.csv").head()