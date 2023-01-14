import pickle

pkl_path = 'prepared_train_annotation.pkl'

data = pickle.load(open(pkl_path,'rb'))
print(data[0])
print(data[0].items())
print(len(data))