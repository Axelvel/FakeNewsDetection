import torch  
import numpy as np
from transformers import BertTokenizer,BertModel
import math

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
model = BertModel.from_pretrained("bert-base-uncased")

#NOTE - Two way to do it:
    # - transform everything to text and add it raw(litteraly add the category's name beofre to create some context)
    # - normalize everything and find a way to sum everything up after embedding

def extract_topic(df):
    topic = df["subject"].to_list()
    nb_topic_max = 0
    all_topic = []
    for line in topic:
        if isinstance(line,str) :
            splitted = line.split(",")
            if len(splitted)>nb_topic_max:
                nb_topic_max = len(splitted)
            for item in splitted:
                all_topic.append(item)
    return list(set(all_topic)),nb_topic_max

def avoid_nan(entry,entry_list):
    try:
        value  = entry_list.index(entry)/len(entry_list)
    except:
        value = 0
    return value

def extract_data(matrix, all_topic, nb_topic_max, all_speaker, all_job, all_states):
    
    statements = []
    meta_datas = []
    mask = []

    for entry in matrix:

        line_meta_datas = []
        
        #convert each subject into a number and normalize it + padd with 0
        if isinstance(entry[2],str):
            for item in entry[2].split(","):
                if nb_topic_max-len(line_meta_datas) > 0 and item in all_topic:
                    line_meta_datas.append(all_topic.index(item)/len(all_topic))
            for i in range(nb_topic_max-len(line_meta_datas)):
                line_meta_datas.append(0.0)

        line_meta_datas.append(avoid_nan(entry[3],all_speaker))
        line_meta_datas.append(avoid_nan(entry[4],all_job))
        line_meta_datas.append(avoid_nan(entry[5],all_states))

        #transform political position into number
        if entry[6] == "republican":
            line_meta_datas.append(-1)
        elif entry[6] == "democrat":
            line_meta_datas.append(1)
        else:
            line_meta_datas.append(0)

        try:
            sum_speech = int(entry[7]) + int(entry[8]) + int(entry[9]) + int(entry[10]) + int(entry[11])
        except:
            sum_speech=0

        for i in range(5):
            if sum_speech != 0:
                line_meta_datas.append(int(entry[7+i])/sum_speech)
            else:
                line_meta_datas.append(0)

        line_meta_datas.append(sum_speech)

        meta_datas.append(line_meta_datas)

        if isinstance(entry[12],str):
            tok =tokenizer(entry[1] + entry[12],max_length=512, add_special_tokens=True, truncation=True, padding='max_length', return_tensors="pt")
        else:
            tok = tokenizer(entry[1],max_length=512, add_special_tokens=True, truncation=True, padding='max_length', return_tensors="pt")
            
        statements.append(tok['input_ids'])
        mask.append(tok['attention_mask'])

    meta_datas = torch.tensor(np.array(meta_datas))

    statements = torch.cat(statements, dim=0)
    mask = torch.cat(mask, dim=0)
    
    return statements,meta_datas,mask


def preprocess(df,all_topic, nb_topic_max,all_speaker,all_job):
    
    df = df.values.tolist()

    with open("./states.txt") as f:
        all_states = f.read()
    all_states = all_states.split()

    return(extract_data(df,all_topic, nb_topic_max,all_speaker,all_job,all_states))