from main import df_train,df_test,df_eval,LABELS
import math

#NOTE - Two way to do it:
    # - transform everything to text and add it raw(litteraly add the category's name beofre to create some context)
    # - normalize everything and find a way to sum everything up after embedding

train_topic = df_train["subject"].to_list()
max = 0
all_topic = []
for line in train_topic:
    if isinstance(line,str) :
        splitted = line.split(",")
        if len(splitted)>max:
            max = len(splitted)
        for item in splitted:
            all_topic.append(item)
all_topic = list(set(all_topic))

all_speaker = list(set(df_train["speaker"].to_list()))
all_job = list(set(df_train["job"].to_list()))

df_train = df_train.values.tolist()
df_eval = df_eval.values.tolist()
df_test = df_test.values.tolist()

#SECTION - text with [SEP]
# ------------------------ PREPROCESSING TRAINING DATA ----------------------- #

# def extract_data(matrix):
#     statement = []
#     meta_datas = []
#     for line in matrix:
#         statement.append([["[CLS]"],[line[1]],["[SEP]"]])

#         meta_datas.append([["[CLS]"],[f"Topic: {line[2]}"],["[SEP]"],[f"Speaker: {line[3]}"]])
    
#     return statement

# print(extract_data(df_train))


#SECTION - Number transformation WIP

with open("./states.txt") as f:
    all_states = f.read()
all_states = all_states.split()

def avoid_nan(entry,entry_list):
    try:
        value  = entry_list.index(entry)/len(entry_list)
    except:
        value = 0
    return value

meta_datas = []
for entry in df_train:

    line_meta_datas = []
    
    #convert each subject into a number and normalize it + padd with 0
    if isinstance(entry[2],str):
        for item in entry[2].split(","):
            line_meta_datas.append(all_topic.index(item)/len(all_topic))
        for i in range(max-len(line_meta_datas)):
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
print(meta_datas)

    
print(df_train[:5])
