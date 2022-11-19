#coding=UTF-8

import ast, re, gc, copy, sys, random, os
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from tqdm.notebook import tqdm
from tqdm import tqdm
from apex import amp

from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizerFast, RobertaModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import lr_scheduler as torch_lrs

BASE_URL = "./data"
N_FOLDS = 5

### Logger setting
import logging
logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s - %(levelname)s -:: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"logger started. 💭BERTa model, KFold={N_FOLDS} 🔴🟡🟢 {sys.argv}")

"""Datasets Helper Function
need to merge features.csv, patient_notes.csv with train.csv
"""
def process_feature_text(text):
    return text.replace("-OR-", ";-").replace("-", " ")

def clean_spaces(text):
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\r', ' ', text)
    return text

def prepare_datasets():
    features = pd.read_csv(f"{BASE_URL}/features.csv")
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"

    notes = pd.read_csv(f"{BASE_URL}/patient_notes.csv")
    df = pd.read_csv(f"{BASE_URL}/train.csv")

    df['annotation'] = df['annotation'].apply(ast.literal_eval)
    df['location'] = df['location'].apply(ast.literal_eval)

    #df["annotation_list"] = [literal_eval(x) for x in df["annotation"]]
    #df["location_list"] = [literal_eval(x) for x in df["location"]]

    ## Using "case_num", "pn_num" to StratifiedKold
    ## NOTE!!!!! USE "patient_notes.csv" first!!!!!
    """
    Fold = StratifiedKFold(n_splits=N_FOLDS, random_state=hyperparameters['seed'], shuffle=True)
    for fold, (_, valid_idx) in enumerate(Fold.split(X=notes , y=notes['case_num'])):
        notes.loc[valid_idx, "fold"] = fold

    counts = notes.groupby(["fold", "pn_num"], as_index=False).count()
    print(counts.shape, counts.pn_num.nunique(), counts.case_num.unique())
    """

    # incorrect annotation
    df.loc[338, 'annotation'] = ast.literal_eval('[["father heart attack"]]')
    df.loc[338, 'location'] = ast.literal_eval('[["764 783"]]')

    df.loc[621, 'annotation'] = ast.literal_eval('[["for the last 2-3 months"]]')
    df.loc[621, 'location'] = ast.literal_eval('[["77 100"]]')

    df.loc[655, 'annotation'] = ast.literal_eval('[["no heat intolerance"], ["no cold intolerance"]]')
    df.loc[655, 'location'] = ast.literal_eval('[["285 292;301 312"], ["285 287;296 312"]]')

    df.loc[1262, 'annotation'] = ast.literal_eval('[["mother thyroid problem"]]')
    df.loc[1262, 'location'] = ast.literal_eval('[["551 557;565 580"]]')

    df.loc[1265, 'annotation'] = ast.literal_eval('[[\'felt like he was going to "pass out"\']]')
    df.loc[1265, 'location'] = ast.literal_eval('[["131 135;181 212"]]')

    df.loc[1396, 'annotation'] = ast.literal_eval('[["stool , with no blood"]]')
    df.loc[1396, 'location'] = ast.literal_eval('[["259 280"]]')

    df.loc[1591, 'annotation'] = ast.literal_eval('[["diarrhoe non blooody"]]')
    df.loc[1591, 'location'] = ast.literal_eval('[["176 184;201 212"]]')

    df.loc[1615, 'annotation'] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
    df.loc[1615, 'location'] = ast.literal_eval('[["249 257;271 288"]]')

    df.loc[1664, 'annotation'] = ast.literal_eval('[["no vaginal discharge"]]')
    df.loc[1664, 'location'] = ast.literal_eval('[["822 824;907 924"]]')

    df.loc[1714, 'annotation'] = ast.literal_eval('[["started about 8-10 hours ago"]]')
    df.loc[1714, 'location'] = ast.literal_eval('[["101 129"]]')

    df.loc[1929, 'annotation'] = ast.literal_eval('[["no blood in the stool"]]')
    df.loc[1929, 'location'] = ast.literal_eval('[["531 539;549 561"]]')

    df.loc[2134, 'annotation'] = ast.literal_eval('[["last sexually active 9 months ago"]]')
    df.loc[2134, 'location'] = ast.literal_eval('[["540 560;581 593"]]')

    df.loc[2191, 'annotation'] = ast.literal_eval('[["right lower quadrant pain"]]')
    df.loc[2191, 'location'] = ast.literal_eval('[["32 57"]]')

    df.loc[2553, 'annotation'] = ast.literal_eval('[["diarrhoea no blood"]]')
    df.loc[2553, 'location'] = ast.literal_eval('[["308 317;376 384"]]')

    df.loc[3124, 'annotation'] = ast.literal_eval('[["sweating"]]')
    df.loc[3124, 'location'] = ast.literal_eval('[["549 557"]]')

    df.loc[3858, 'annotation'] = ast.literal_eval('[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]')
    df.loc[3858, 'location'] = ast.literal_eval('[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]')

    df.loc[4373, 'annotation'] = ast.literal_eval('[["for 2 months"]]')
    df.loc[4373, 'location'] = ast.literal_eval('[["33 45"]]')

    df.loc[4763, 'annotation'] = ast.literal_eval('[["35 year old"]]')
    df.loc[4763, 'location'] = ast.literal_eval('[["5 16"]]')

    df.loc[4782, 'annotation'] = ast.literal_eval('[["darker brown stools"]]')
    df.loc[4782, 'location'] = ast.literal_eval('[["175 194"]]')

    df.loc[4908, 'annotation'] = ast.literal_eval('[["uncle with peptic ulcer"]]')
    df.loc[4908, 'location'] = ast.literal_eval('[["700 723"]]')

    df.loc[6016, 'annotation'] = ast.literal_eval('[["difficulty falling asleep"]]')
    df.loc[6016, 'location'] = ast.literal_eval('[["225 250"]]')

    df.loc[6192, 'annotation'] = ast.literal_eval('[["helps to take care of aging mother and in-laws"]]')
    df.loc[6192, 'location'] = ast.literal_eval('[["197 218;236 260"]]')

    df.loc[6380, 'annotation'] = ast.literal_eval('[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]')
    df.loc[6380, 'location'] = ast.literal_eval('[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]')

    df.loc[6562, 'annotation'] = ast.literal_eval('[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]')
    df.loc[6562, 'location'] = ast.literal_eval('[["290 320;327 337"], ["290 320;342 358"]]')

    df.loc[6862, 'annotation'] = ast.literal_eval('[["stressor taking care of many sick family members"]]')
    df.loc[6862, 'location'] = ast.literal_eval('[["288 296;324 363"]]')

    df.loc[7022, 'annotation'] = ast.literal_eval('[["heart started racing and felt numbness for the 1st time in her finger tips"]]')
    df.loc[7022, 'location'] = ast.literal_eval('[["108 182"]]')

    df.loc[7422, 'annotation'] = ast.literal_eval('[["first started 5 yrs"]]')
    df.loc[7422, 'location'] = ast.literal_eval('[["102 121"]]')

    df.loc[8876, 'annotation'] = ast.literal_eval('[["No shortness of breath"]]')
    df.loc[8876, 'location'] = ast.literal_eval('[["481 483;533 552"]]')

    df.loc[9027, 'annotation'] = ast.literal_eval('[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]')
    df.loc[9027, 'location'] = ast.literal_eval('[["92 102"], ["123 164"]]')

    df.loc[9938, 'annotation'] = ast.literal_eval('[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]')
    df.loc[9938, 'location'] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

    df.loc[9973, 'annotation'] = ast.literal_eval('[["gaining 10-15 lbs"]]')
    df.loc[9973, 'location'] = ast.literal_eval('[["344 361"]]')

    df.loc[10513, 'annotation'] = ast.literal_eval('[["weight gain"], ["gain of 10-16lbs"]]')
    df.loc[10513, 'location'] = ast.literal_eval('[["600 611"], ["607 623"]]')

    df.loc[11551, 'annotation'] = ast.literal_eval('[["seeing her son knows are not real"]]')
    df.loc[11551, 'location'] = ast.literal_eval('[["386 400;443 461"]]')

    df.loc[11677, 'annotation'] = ast.literal_eval('[["saw him once in the kitchen after he died"]]')
    df.loc[11677, 'location'] = ast.literal_eval('[["160 201"]]')

    df.loc[12124, 'annotation'] = ast.literal_eval('[["tried Ambien but it didnt work"]]')
    df.loc[12124, 'location'] = ast.literal_eval('[["325 337;349 366"]]')

    df.loc[12279, 'annotation'] = ast.literal_eval('[["heard what she described as a party later than evening these things did not actually happen"]]')
    df.loc[12279, 'location'] = ast.literal_eval('[["405 459;488 524"]]')

    df.loc[12289, 'annotation'] = ast.literal_eval('[["experienced seeing her son at the kitchen table these things did not actually happen"]]')
    df.loc[12289, 'location'] = ast.literal_eval('[["353 400;488 524"]]')

    df.loc[13238, 'annotation'] = ast.literal_eval('[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
    df.loc[13238, 'location'] = ast.literal_eval('[["293 307"], ["321 331"]]')

    df.loc[13297, 'annotation'] = ast.literal_eval('[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]')
    df.loc[13297, 'location'] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

    df.loc[13299, 'annotation'] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
    df.loc[13299, 'location'] = ast.literal_eval('[["79 88"], ["409 418"]]')

    df.loc[13845, 'annotation'] = ast.literal_eval('[["headache global"], ["headache throughout her head"]]')
    df.loc[13845, 'location'] = ast.literal_eval('[["86 94;230 236"], ["86 94;237 256"]]')

    df.loc[14083, 'annotation'] = ast.literal_eval('[["headache generalized in her head"]]')
    df.loc[14083, 'location'] = ast.literal_eval('[["56 64;156 179"]]')

    merged = df.merge(notes, how="left")
    merged = merged.merge(features, how="left")

    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    merged["feature_text"] = merged["feature_text"].apply(clean_spaces)
    ### merged["feature_text"] = merged["feature_text"].apply(lambda x: x.lower())

    merged["pn_history"] = merged["pn_history"].apply(lambda x: x.strip())
    merged["pn_history"] = merged["pn_history"].apply(clean_spaces)
    ### merged["pn_history"] = merged["pn_history"].apply(lambda x: x.lower())

    
    ## GroupKFold
    Fold = GroupKFold(n_splits=N_FOLDS)
    groups = merged['pn_num'].values
    merged["fold"] = -1
    for fold, (_, valid_idx) in enumerate(Fold.split(merged, merged['location'], groups)):
        merged.loc[valid_idx, "fold"] = fold
       

    return merged

#Tokenizer Helper Function
def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return

def tokenize_and_add_labels(tokenizer, data, config):
    out = tokenizer(
        data["feature_text"],
        data["pn_history"],
        truncation=config['truncation'],
        max_length=config['max_length'],
        padding=config['padding'],
        return_offsets_mapping=config['return_offsets_mapping']
    )
    labels = [0.0] * len(out["input_ids"])
    out["location_int"] = loc_list_to_ints(data["location"])
    out["sequence_ids"] = out.sequence_ids()

    for idx, (seq_id, offsets) in enumerate(zip(out["sequence_ids"], out["offset_mapping"])):
        if not seq_id or seq_id == 0:
            labels[idx] = -1
            continue

        exit = False
        token_start, token_end = offsets
        for feature_start, feature_end in out["location_int"]:
            if exit:
                break
            ##if token_start >= feature_start and token_end <= feature_end:
            if token_start <= feature_start < token_end or token_start < feature_end <= token_end or feature_start <= token_start < feature_end:
                labels[idx] = 1.0
                exit = True

    out["labels"] = labels

    return out


#Predection and Score Helper Function
from sklearn.metrics import accuracy_score

def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    all_predictions_test = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        pred = 1 / (1 + np.exp(-pred))
        start_idx = None
        end_idx = None
        current_preds = []
        current_preds_test = []
        for pred, offset, seq_id in zip(pred, offsets, seq_ids):
            if seq_id is None or seq_id == 0:
                continue

            if pred > 0.5:
                if start_idx is None:
                    start_idx = offset[0]
                end_idx = offset[1]
            elif start_idx is not None:
                current_preds_test.append(f"{start_idx} {end_idx}")
                current_preds.append((start_idx, end_idx))
                start_idx = None

        all_predictions_test.append("; ".join(current_preds_test))
        all_predictions.append(current_preds)
            
    return all_predictions, all_predictions_test

def calculate_char_cv(predictions, offset_mapping, sequence_ids, labels, texts):
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels, text in zip(predictions, offset_mapping, sequence_ids, labels, texts["pn_history"]):

        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros(num_chars)

        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0]:o[1]] = 1
                if text[o[0]].isspace() and o[0]>0 and char_labels[o[0]-1]!=1:
                    char_labels[o[0]] = 0


        char_preds = np.zeros(num_chars)

        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
            if text[start_idx].isspace():
                char_preds[start_idx] = 0

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    results = precision_recall_fscore_support(all_labels, all_preds, average="binary", labels=np.unique(all_preds))
    accuracy = accuracy_score(all_labels, all_preds)
    

    return {
        "Accuracy": accuracy,
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }

### Dataset for DataLoader
class CustomDataset(Dataset):
    def __init__ (self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__ (self):
        return len(self.data)

    def __getitem__ (self, idx):
        data = self.data.iloc[idx]
        tokens = tokenize_and_add_labels(self.tokenizer, data, self.config)
        
        input_ids = np.array(tokens["input_ids"])
        attention_mask = np.array(tokens["attention_mask"])
        ### token_type_ids = np.array(tokens["token_type_ids"])

        labels = np.array(tokens["labels"])
        offset_mapping = np.array(tokens['offset_mapping'])
        sequence_ids = np.array(tokens['sequence_ids']).astype("float16")
        
        ### return input_ids, attention_mask, token_type_ids, labels, offset_mapping, sequence_ids
        return input_ids, attention_mask, labels, offset_mapping, sequence_ids

"""
Model
Lets use BERT base Architecture
Also Used 3 FC layers
Comments: 3 layers improve accuracy 2% on public score
"""
from DeBERTa_RTD.DeBERTa.apps.models.replaced_token_detection_model import ReplacedTokenDetectionModel
RTD_model_path = "/home/xyb/Project/Kaggle/NBME/model/huggingface-bert/microsoft/deberta-v2-xlarge/pytorch_model.bin"
RTD_model_cfg = "/home/xyb/Project/Kaggle/NBME/DeBERTa_RTD/experiments/language_model/deberta_xlarge.json"
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.modelCfg = AutoConfig.from_pretrained(config['model_name'], output_hidden_states=True)
        if self.config['RTDM']:
            self.fc = nn.Linear(512, 1)
        else:
            self.fc = nn.Linear(self.modelCfg.hidden_size, 1)
        if "deberta" in config['model_name']:
            if self.config['RTDM']:
                self.model = ReplacedTokenDetectionModel.load_model(model_path=RTD_model_path, model_config=RTD_model_cfg)
            else:
                self.model = AutoModel.from_pretrained(config['model_name'], config=self.modelCfg)  # DeBERTa Model
                self._init_weights(self.fc)

        elif "roberta" in config['model_name']:
            self.model = RobertaModel.from_pretrained(config['model_name'])  # RoBERTa model
        else:
            self.model = AutoModel.from_pretrained(config['model_name'])  # BERT model
        self.current_epoch = -1

        if 'deberta-v2-xxlarge' in config['model_name']:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in config['model_name']:
            #self.model.embeddings.requires_grad_(False)
            #self.model.encoder.layer[:12].requires_grad_(False) # 冻结12/24
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

    def forward(self, input_ids, attention_mask, labels):
        if self.config['RTDM']:
            outputs = self.model(input_ids=input_ids, input_mask=attention_mask, labels=labels)
            logits = outputs['logits']
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.fc(self.dropout(outputs[0])).squeeze(-1)        
        return logits


###Hyperparameters
hyperparameters = {
    "max_length": 512, ### Seems that the max lenth should be 466 not 416
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "dropout": 0.2,
    "scheduler": "cosine", # ['linear', 'cosine', 'cosine_anneal_restart', polynomial']
    "batch_scheduler": True,
    "num_warmup_steps": 0, ##warmup epoch number, 0==>No Warmup.
    "num_cycles": 0.5,
    "lr": 2e-6, ### good for all others :1e-5, except Dv2XL should use  2e-6,
    "min_lr": 1e-7,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "seed": 1979,
    "batch_size": 1, ### without [@torch.no_grad() + AMP], the value was 3. Now, 8!
    "gradient_accumulation_steps": 1,
    "epochs": 4,
    "model_name": "TBD",
    "model_prefix": "TBD",
    "enable_amp_half_precision": False,
    "RTDM": False, ### replaced token detection, RTD model
}

model_list = {
    "deberta": "./model/huggingface-bert/microsoft/deberta-v3-large", #batchsize:: AMP 8 / NonAMP 3 / RTD 4
    #"deberta": "./deberta-v3-large-regular", #batchsize:: AMP 8 / NonAMP 3 / RTD 4
    "deberta_desc": "DeBerta-V3-Large",

    #"dbtV2XL": "./model/huggingface-bert/microsoft/deberta-v2-xlarge", #batchsize:: after freeze[:12] AMP 4 / NonAMP 3
    "dbtV2XL": "./deberta-v2-xlarge-regular/", #batchsize:: after freeze[:12] AMP 4 / NonAMP 3
    "dbtV2XL_desc": "DeBerta-V2-XLarge",                               # RTD 1 (no freeze)

    "dbtV2XXL": "./model/huggingface-bert/microsoft/deberta-v2-xxlarge", #batchsize:: after freeze[:24] AMP :( / NonAMP :(
    "dbtV2XXL_desc": "DeBerta-V2-XXLarge",

    "roberta": "./model/huggingface-bert/roberta-large",
    "roberta_desc": "RoBerta-Large",

    "bert": "./model/huggingface-bert/bert-base-uncased",
    "bert_desc": "Bert-Base-Uncased"
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
#seed_everything(hyperparameters['seed'])

def train_model(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    train_loss = []

    pbar = tqdm(dataloader)
    step = 0
    total_steps = len(dataloader)
    optimizer.zero_grad()
    for batch in pbar:
        step += 1
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        ### token_type_ids = batch[2].to(DEVICE)
        ### labels = batch[3].to(DEVICE)
        labels = batch[2].to(DEVICE)

        ### logits = model(input_ids, attention_mask, token_type_ids)
        logits = model(input_ids, attention_mask, labels)
        loss = criterion(logits, labels)
        # since, we have
        loss = torch.masked_select(loss, labels > -1.0).mean()
        if hyperparameters["gradient_accumulation_steps"] > 1:
            loss = loss / hyperparameters["gradient_accumulation_steps"]
        train_loss.append(loss.item() * input_ids.size(0))

        if (hyperparameters['enable_amp_half_precision'] == True):
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # it's also improve f1 accuracy slightly
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if step % hyperparameters["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            if hyperparameters['batch_scheduler']:
                if hyperparameters['scheduler'] == "cosine_anneal_restart":
                    scheduler.step(model.current_epoch + step / total_steps)
                else:
                    scheduler.step()

        ##lr = scheduler.get_last_lr()[0]
        ##print(f"LR: {lr:.8f}〽️")
        pbar.set_postfix(Epoch=f"{model.current_epoch+1}/{hyperparameters['epochs']}", Train_Loss=(sum(train_loss)/len(train_loss)),
                        LR=scheduler.get_last_lr()[0])
        pbar.update()
        del input_ids, attention_mask, labels, logits

    lr = scheduler.get_last_lr()[0]
    print(f"LR: {lr:.8f}〽️")
    logger.info(f"LR: {lr:.8f}〽️")
    pbar.close()
    gc.collect()
    torch.cuda.empty_cache()

    return sum(train_loss)/len(train_loss)

@torch.no_grad()
def eval_model(model, dataloader, criterion, texts):
    model.eval()
    valid_loss = []
    preds = []
    offsets = []
    seq_ids = []
    valid_labels = []

    pbar = tqdm(dataloader)

    for batch in pbar:
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        ### token_type_ids = batch[2].to(DEVICE)
        ### labels = batch[3].to(DEVICE)
        labels = batch[2].to(DEVICE)
        offset_mapping = batch[3]
        sequence_ids = batch[4]

        ### logits = model(input_ids, attention_mask, token_type_ids)
        logits = model(input_ids, attention_mask, labels)
        loss = criterion(logits, labels)
        loss = torch.masked_select(loss, labels > -1.0).mean()
        ##if hyperparameters["gradient_accumulation_steps"] > 1:
        ##   loss = loss / hyperparameters["gradient_accumulation_steps"]
        valid_loss.append(loss.item() * input_ids.size(0))

        preds.append(logits.detach().cpu().numpy())
        offsets.append(offset_mapping.numpy())
        seq_ids.append(sequence_ids.numpy())
        valid_labels.append(labels.detach().cpu().numpy())
        pbar.set_postfix(Epoch=f"{model.current_epoch+1}/{hyperparameters['epochs']}", Valid_Loss=(sum(valid_loss)/len(valid_loss)))

        del input_ids, attention_mask, labels, sequence_ids, logits

    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    location_preds, location_preds_test = get_location_predictions(preds, offsets, seq_ids, test=False)
    score = calculate_char_cv(location_preds, offsets, seq_ids, valid_labels, texts)

    gc.collect()
    torch.cuda.empty_cache()

    return sum(valid_loss)/len(valid_loss), score, location_preds, location_preds_test

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg['scheduler'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=(num_train_steps / cfg["epochs"] * cfg["num_warmup_steps"]), num_training_steps=num_train_steps
        )
    elif cfg['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=(num_train_steps / cfg["epochs"] * cfg["num_warmup_steps"]), num_training_steps=num_train_steps, num_cycles=cfg["num_cycles"]
        )
    elif cfg['scheduler'] == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=(num_train_steps / cfg["epochs"] * cfg["num_warmup_steps"]), num_training_steps=num_train_steps, lr_end=cfg["min_lr"], power=3
        )
    elif cfg['scheduler'] == 'cosine_anneal_restart':
        scheduler = torch_lrs.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=1, eta_min=cfg["min_lr"]
        )

    return scheduler

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

DEVICE = "cuda"
train_df = prepare_datasets()

print (f"parameter_number=={len(sys.argv)}")
if len(sys.argv) != 2:
        print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} d' / '{sys.argv[0]} r' to use 'DeBERTa' or 'RoBERTa' model.")
        sys.exit()
elif sys.argv[1] != "d" and sys.argv[1] != "r" and sys.argv[1] != 'b' and sys.argv[1] != 'x' and sys.argv[1] != 'xx':
        print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} d' / '{sys.argv[0]} r' to use 'DeBERTa' or 'RoBERTa' model.")
        sys.exit()
else:
    if sys.argv[1] == "d":
        hyperparameters["model_name"] = model_list["deberta"]
        hyperparameters["model_prefix"] = model_list["deberta_desc"]
    elif sys.argv[1] == "r":
        hyperparameters["model_name"] = model_list["roberta"]
        hyperparameters["model_prefix"] = model_list["roberta_desc"]
    elif sys.argv[1] == "b":
        hyperparameters["model_name"] = model_list["bert"]
        hyperparameters["model_prefix"] = model_list["bert_desc"]
    elif sys.argv[1] == "x":
        hyperparameters["model_name"] = model_list["dbtV2XL"]
        hyperparameters["model_prefix"] = model_list["dbtV2XL_desc"]
    elif sys.argv[1] == "xx":
        hyperparameters["model_name"] = model_list["dbtV2XXL"]
        hyperparameters["model_prefix"] = model_list["dbtV2XXL_desc"]

from transformers.models.deberta_v2 import DebertaV2TokenizerFast
if "deberta" in hyperparameters["model_name"]:
    tokenizer = DebertaV2TokenizerFast.from_pretrained(hyperparameters["model_name"])
elif "roberta" in hyperparameters["model_name"]:
    tokenizer = RobertaTokenizerFast.from_pretrained(hyperparameters["model_name"])
else:
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"])


"""
tokenizer_1 = RobertaTokenizerFast.from_pretrained(model_list['roberta'], trim_offsets=False)
tokenizer_2 = RobertaTokenizerFast.from_pretrained(model_list['roberta'], trim_offsets=True)

tmp = 'patient with recent heart attack'

encode_1 = tokenizer_1(tmp, 
                    return_offsets_mapping=True)

encode_2 = tokenizer_2(tmp, 
                    return_offsets_mapping=True)

for (start,end) in encode_1['offset_mapping']:
    print(f"'{tmp[start:end]}', {start}, {end}")
print(f"-------------------------")
for (start,end) in encode_2['offset_mapping']:
    print(f"'{tmp[start:end]}', {start}, {end}")

print (f"RoBERTa Tokenizer test ends.")
"""

def train_K_Fold(fold, model, optimizer):
    """
    Prepare Datasets
    """
    print(f"*************************\nRuning fold==>{fold+1}/{N_FOLDS}\n*************************")
    logger.info(f"*************************\nRuning fold==>{fold+1}/{N_FOLDS} 🌱🌱🌱⚡️🍄🍄🍄")
    X_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
    X_test = train_df[train_df["fold"] == fold].reset_index(drop=True)
    print ("TrainSet size=={}, TestSet size=={}\n*************************".format(len(X_train), len(X_test)))
    logger.info("TrainSet size=={}, TestSet size=={}".format(len(X_train), len(X_test)))

    training_data = CustomDataset(X_train, tokenizer, hyperparameters)
    train_dataloader = DataLoader(training_data, batch_size=hyperparameters['batch_size'], shuffle=True)

    test_data = CustomDataset(X_test, tokenizer, hyperparameters)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters['batch_size'], shuffle=False)

    """
    Train
    Lets train the model with BCEWithLogitsLoss and AdamW as optimizer

    Notes: on BCEWithLogitsLoss, the default value for reductio
    n is mean (the sum of the output will be divided by the number of elements in the output). 
    If we use this default value, it will produce negative loss. 
    Because we have some negative labels. 
    To fix this negative loss issue, we can use none as parameter. 
    To calculate the mean, first, we have to filter out the negative values.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    """

    epochs = hyperparameters['epochs']

    #model = CustomModel(hyperparameters).to(DEVICE)
    checkpoint = torch.load("./model_initialized.bin")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    """
    optimizer_parameters = get_optimizer_params(model,
                                            encoder_lr=hyperparameters['lr'], 
                                            decoder_lr=hyperparameters['lr'],
                                            weight_decay=hyperparameters['weight_decay'])
    """
    criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")
    #optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
    ###optimizer = optim.AdamW(optimizer_parameters, lr=hyperparameters['lr'])

    num_train_steps = int(len(X_train) / (hyperparameters['batch_size'] * hyperparameters["gradient_accumulation_steps"]) * epochs)
    scheduler = get_scheduler(hyperparameters, optimizer, num_train_steps)

    train_loss_data, valid_loss_data = [], []
    score_data_list = []
    valid_loss_min = np.Inf
    
    best_loss = np.inf
    best_score = 0

    for i in range(epochs):
        epoch_since = time.time()
        print("Epoch: {}/{}".format(i + 1, epochs))
        logger.info("Epoch: {}/{}".format(i + 1, epochs))

        # first train model
        model.current_epoch = i
        train_loss = train_model(model, train_dataloader, optimizer, criterion, scheduler)
        train_loss_data.append(train_loss)
        print(f"Train loss: {train_loss}")
        logger.info(f"Train loss: {train_loss}")
        # then evaluate model
        valid_loss, score, location_preds, location_preds_test = eval_model(model, test_dataloader, criterion, X_test)
        valid_loss_data.append(valid_loss)
        score_data_list.append(score)
        print(f"Valid loss: {valid_loss}")
        print(f"Valid score: {score}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"nbme_{hyperparameters['model_prefix']}_fold-{fold}-BV.pth")
            logger.info(f"Valid loss: {valid_loss} --- !!!Fine-tuning UPDATED!!!🥰👽😽")
            #torch.save(model, "nbme_bert_model.model")
        else:
            logger.info(f"Valid loss: {valid_loss}")

        if score['f1'] > best_score:
            best_score = score['f1']
            torch.save(model.state_dict(), f"nbme_{hyperparameters['model_prefix']}_fold-{fold}-BS.pth")
            logger.info(f"💯🆕Valid score: {score}")
        else:
            logger.info(f"Valid score: {score}")
            
        
        epoch_time_elapsed = time.time() - epoch_since
        logger.info('Epoch: {} completed in {:.0f}m {:.0f}s'.format(i+1, epoch_time_elapsed // 60, epoch_time_elapsed % 60))

        X_test["location2"] = location_preds
        X_test["location_test"] = location_preds_test
        X_test[["id", "location"]].to_csv(f"submission-T-{i+1}.csv", index = False)
        X_test[["id", "location_test"]].to_csv(f"submission-F-{i+1}.csv", index = False)


"""
MAIN FUNC
"""
import time
if __name__=='__main__':

    since = time.time()

    model = CustomModel(hyperparameters).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
    #optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparameters['lr'])

    if (hyperparameters['enable_amp_half_precision'] == True):
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, "./model_initialized.bin")

    for fold in range(N_FOLDS):
        train_K_Fold(fold, model, optimizer)
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


