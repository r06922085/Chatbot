#"mode con cp select=950"
from utils.dataManager import DataManager
from models.Model import Model
import tensorflow as tf
import numpy as np
import os
import argparse
import math

class main(object):
    def parse(self):
        parser = argparse.ArgumentParser(description="chatbot")
        parser.add_argument('--train', action='store_true', help='whether train')
        parser.add_argument('--test', action='store_true', help='whether test')
        parser.add_argument('--model_restore', action='store_true', help='whether restore the model')
        
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
        self.val_size,self.train_x,self.train_y,self.dictionary = dataset.getTrainData()
        tf.reset_default_graph()
    def set_testing_data(self):
        print('setting testing data...')
        self.max_len = 25
        dataset = DataManager(max_len = self.max_len)
        self.val_size, self.dictionary = dataset.getTestData()
        tf.reset_default_graph()
    def getting_model(self, args):
        print('getting model...')
        if args.train or args.test:
            self.model = Model(batch_size = self.batch_size , val_size = self.val_size, 
                max_len = self.max_len ,args = args , dictionary = self.dictionary)#
            self.model.compile()
            if (args.train and args.model_restore) or args.test:
                self.model.restore(mode = self.mode)
    
    def train(self, args):
        print('start traing...')
        #start train
        epoch = 0
        min_loss = math.inf
        while True:
            epoch += 1
            loss = self.model.fit(self.train_x, self.train_y, self.batch_size, epoch)
            
            #store the Model
            self.model.save()
        
    def test(self, args):  
        print('start testing...')     
        while True:
            ques = input('請說話...')
          
            ans = self.model.predict(ques)
            
            print(ans)
            
        
if __name__ == '__main__':
    
    chatbot = main()
    
    args = chatbot.parse()
    chatbot.set_parameter(args)
    
    #start traing or testing
    if args.train:
        chatbot.set_training_data()
        chatbot.getting_model(args)
        chatbot.train(args)
    elif args.test:
        chatbot.set_testing_data()
        chatbot.getting_model(args)
        chatbot.test(args)
    
    

            

