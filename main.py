#"mode con cp select=950"
from utils.dataManager import DataManager
from models.hw2_1_S2VT import Seq2Seq
import tensorflow as tf
import numpy as np
import os
import listen_speak as ls
import argparse

class main(object):
    def parse(self):
        parser = argparse.ArgumentParser(description="chatbot")
        parser.add_argument('--train', action='store_true', help='whether train')
        parser.add_argument('--test', action='store_true', help='whether test')
        parser.add_argument('--text', action='store_true', help='whether test')
        parser.add_argument('--model_restore', action='store_true', help='whether restore the model')
        parser.add_argument('--ques_from_text', action='store_true', 
            help='whether test questions come from a text file')
        try:
            from argument import add_arguments
            parser = add_arguments(parser)
        except:
            pass
        args = parser.parse_args()
        return args
    def set_parameter(self, args):
        print('setting parameters...')
        if args.train:
            self.batch_size = 200
            self.mode = 'train'
        elif args.test:
            self.batch_size = 1
            self.mode = 'test'
    
    def set_training_data(self):
        print('getting training data...')
        self.max_len = 25
        dataset = DataManager(max_len = self.max_len)
        self.voc_size,self.train_x,self.train_y,self.dictionary = dataset.getTrainData()
        tf.reset_default_graph()
    def set_testing_data(self):
        print('setting testing data...')
        self.max_len = 25
        dataset = DataManager(max_len = self.max_len)
        self.voc_size, self.dictionary = dataset.getTestData()
        tf.reset_default_graph()
    def getting_model(self, args):
        print('getting model...')
        if args.train or args.test:
            self.model = Seq2Seq(batch_size = self.batch_size , voc_size = self.voc_size, max_len = self.max_len , 
                mode = self.mode , dictionary = self.dictionary , train_by_correct_input = True)#
            self.model.compile()
            if (args.train and args.model_restore) or args.test:
                self.model.restore(mode = self.mode)
    def start_testing_or_training(self, args):  
        print('start test or training...')     
        if args.test:
            if args.ques_from_text:
                with open('question.txt','r',encoding='utf8') as f:
                    ques = f.read().split('\n')
                f = open('ans.txt','w',encoding='utf8')
                for i in ques:
                    q = i.replace(' ','')
                    ans = self.model.predict(q)
                    f.write(q)
                    f.write('\n')
                    f.write(ans)
                    f.write('\n')
                    f.write('\n')
                    print('done')
            else:
                if args.text:
                    while True:
                        ques = input('請說話...')
                        print(ques)
                        print('回答中...')
                        ans = self.model.predict(ques)
                        ls.speak(ans)
                        print(ans,'\n')
                else:
                    while(True):
                        print('請說話...')
                        ques = ls.listen()
                        print(ques)
                        print('回答中...')
                        ans = self.model.predict(ques)
                        ls.speak(ans)
                        print(ans,'\n')
            
        elif args.train: 
            epoch = 0
            while True:
                epoch += 1
                print('epoch', epoch)
                self.model.fit(self.train_x, self.train_y, self.batch_size, epoch)
                self.model.save()
        #end
        print('finish')
if __name__ == '__main__':
    chatbot = main()
    args = chatbot.parse()
    chatbot.set_parameter(args)
    if args.train:
        chatbot.set_training_data()
    elif args.test:
        chatbot.set_testing_data()
    chatbot.getting_model(args)
    chatbot.start_testing_or_training(args)
    

            

