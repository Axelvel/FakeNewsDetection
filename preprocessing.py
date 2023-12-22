from main import df_train,df_test,df_eval,labels

#NOTE - Two way to do it:
    # - transform everything to text and add it raw(litteraly add the category's name beofre to create some context)
    # - normalize everything and find a way to sum everything up after embedding

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
    states = f.read()
states = states.split()

for i,entry in enumerate(df_train):

    #Transform label name into number
    df_train[i][0] = labels.index(entry[0])

    #transform state into a number
    try:
        df_train[i][5] = states.index(entry[5])/len(states)
    except:
        df_train[i][5] = 0

    #transform political position into number
    if entry[6] == "republican":
        df_train[i][6] = -1
    elif entry[6] == "democrat":
        df_train[i][6] = 1
    else:
        df_train[i][6] = 0

    
print(df_train[:5])
