import numpy as np
import json
import pickle
import json

PAD = '<PAD>' # index 0
BOS = '<BOS>' # index 1
EOS = '<EOS>' # index 2
UNK = '<UNK>' # index 3

        
class DataManager():
    raw_data = None
    whole_data = None
    train_data = None
    train_label = None
    clean_train_label = None
    
    #word_list
    word_list = [PAD, BOS, EOS, UNK]
    word_list_final = [PAD, BOS, EOS, UNK]
    dictionary = []
    times = [0,0,0,0]
    voc_size = 4
    max_len = 0
    file_name = ''
    # data used for training
    train_x = []
    train_y = []
    def __init__(self, max_len, file_name = 'utils/data/clr_conversation.txt'):
        self.max_len = max_len
        self.file_name = file_name
        do = "nothing"
    def getTrainData(self):
        self.get_word_list_final_from_word_list_file()
        self.read_npy_train_data()
        return self.voc_size,self.train_x,self.train_y,self.word_list_final
    def getTestData(self):
        word_list = self.get_word_list_final_from_word_list_file()
        voc_size = self.get_voc_size(word_list)
        return voc_size, word_list
        
    def LoadData(self, file_name = 'utils/data/clr_conversation.txt'):

        # data, label
        train_data = []
        train_label = []
        whole_data = []

        # load file id
        with open(file_name, 'r', encoding = 'utf8',errors='ignore') as f:
            #f.encoding = 'utf8'
            train_sentences = f.read().split('\n')
        for i in range(len(train_sentences)-1):
            #print(train_sentences[i])
            if train_sentences[i] != '+++$+++' and train_sentences[i+1] != '+++$+++':
                train_data.append(train_sentences[i].split())
                train_label.append(train_sentences[i+1].split())
                
        whole_data = [] + train_data#[] is to not to let the train_data address point to whole_data address
        whole_data.append(train_label[-1])        
        self.train_data = train_data
        self.train_label = train_label
        self.whole_data = whole_data
    def get_word_list_final_from_word_list_file(self):
        with open('utils/data/dictionary.txt', 'rb') as f:
            self.word_list_final = pickle.load(f)
        return self.word_list_final
    def get_voc_size(self, word_list):
        self.voc_size = len(word_list)
        return self.voc_size
    def add_words(self):
        for i in range(len(self.whole_data)):
            if (i%10000) == 0:
                print(i,'/',len(self.whole_data),'it is add_words function!')
            for ii in range(len(self.whole_data[i])):
                if(self.whole_data[i][ii] not in self.word_list):
                    self.word_list.append(self.whole_data[i][ii])
                    self.times.append(1)
                else:
                    index = self.word_list.index(self.whole_data[i][ii])
                    self.times[index] += 1 
                    if self.times[index] == 50:
                        self.word_list_final.append(self.whole_data[i][ii])
                    #print(len(self.word_list))
        self.voc_size = len(self.word_list_final)
        self.store_word_list()
    def BuildTrainableData(self):
        self.get_word_list_final_from_word_list_file()
        max_len = self.max_len
        data_len = len(self.train_data)# about 2.7 millions...
        print('data_len',data_len)
        train_x = np.zeros((data_len,max_len))
        train_y = np.zeros((data_len,max_len))
        for i in range(data_len):#build data
            if (i%10000) == 0:
                print(i,'/',data_len,'it is BuildTrainableData!')
            for ii in range(max_len):          
                if ii < len(self.train_data[i]):#not padding
                    if self.train_data[i][ii] in self.word_list_final:
                        index = self.word_list_final.index(self.train_data[i][ii])
                        train_x[i][ii] = index
                    else:
                        train_x[i][ii] = 3
                elif ii == len(self.train_data[i]):
                    train_x[i][ii] = 2
                else:#padding
                    train_x[i][ii] = 0
        for i in range(data_len):#build label
            if (i%10000) == 0:
                print(i,'/',data_len,'it is BuildTrainableData!')
            for ii in range(max_len):          
                if ii < len(self.train_label[i]):#not padding
                    if self.train_label[i][ii] in self.word_list_final:
                        index = self.word_list_final.index(self.train_label[i][ii])
                        train_y[i][ii] = index
                    else:
                        train_y[i][ii] = 3
                elif ii == len(self.train_label[i]):
                    train_y[i][ii] = 2
                else:#padding
                    train_y[i][ii] = 0  
        self.train_x = train_x
        self.train_y = train_y
        self.store_train_data()
    def get_data_word(self,index):
        data_word = []
        for i in range(self.voc_size):
            if i == index:
                data_word.append(1)
            else:
                data_word.append(0)
        return data_word
    def read_train_data(self):
        self.get_voc_size()
        with open('utils/data/train_data_x.txt', 'rb') as f:
            self.train_x = pickle.load(f)
        with open('utils/data/train_data_y.txt', 'rb') as f:
            self.train_y = pickle.load(f)
    def read_npy_train_data(self):
        self.voc_size = 3004
        self.train_x = np.load('utils/data/train_data.npy')
        self.train_y = np.load('utils/data/train_label.npy')
        self.train_y = self.train_y[:,1:]
    def store_train_data(self):
        with open('utils/data/train_data_x(25).txt', 'wb') as f_x:
            pickle.dump(self.train_x,f_x)
        with open('utils/data/train_data_y(25).txt', 'wb') as f_y:
            pickle.dump(self.train_y,f_y)
    def store_word_list(self):
        with open('utils/data/word_list.txt', 'wb') as f:
            pickle.dump(self.word_list_final,f)