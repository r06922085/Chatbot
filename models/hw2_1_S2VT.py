import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple, MultiRNNCell
import os

"""Usage
Declare a Seq2Seq instance : model = Seq2Seq(vac_size)
Compile model : model.compile()
Train model : model.fit(xs, ys, batch_size, epoch)
"""
class Seq2Seq():
    def __init__(self, batch_size , voc_size, max_len, mode , dictionary , train_by_correct_input , dtype = tf.float32):
        # model parameter
        self.dtype = dtype
        self.encoder_units = 512
        self.decoder_units = 1024
        self.max_len = max_len
        self.mode = mode
        self.train_by_correct_input = train_by_correct_input
        self.dictionary = dictionary
        self.voc_size = voc_size
        self.encoder_lay_Num = 3
        self.decoder_lay_Num = 3
        
        # model batch size (a int tensor)
        self.batch_size = batch_size
        
        # feed tensor
        self.xs = tf.placeholder(dtype = self.dtype, shape = [self.batch_size, self.max_len])
        self.ys = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.max_len]) # label : ex.[1, 4, 5, 6, 200]
        self.inputs_length = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        self.inputs_length_test = tf.placeholder(dtype = tf.int32, shape = [1])
        
        # model iporttant tensor (will be initialize by calling compile)
        self.encoder_output = None
        self.attention_output = None
        self.decoder_output = None
        self.decoder_loss = None
        self.train_op = None
        self.prediction = None
        self.print_code = None
        
        # for transform decoder output from dimension 1024 >> 512
        self.decoder_W = None
        self.decoder_b = None
        
        # embeddings
        self.emb_W = None
        self.emb_b = None
        
        # word_embbding
        self.wordEmb_W = None
        self.wordEmb_b = None
        
        #output_embedding
        self.outputEmb_W = None
        self.outputEmb_b = None
        
        # training data
        self.train_x = None
        self.train_y = None
        
        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)  
        
        
    def compile(self):
        # initialize emb transform
        self.emb_W = tf.Variable(tf.truncated_normal([self.decoder_units // 2, self.voc_size], 
                                                     stddev=1.0 / np.sqrt(self.encoder_units), dtype = self.dtype))
        self.emb_b = tf.Variable(tf.zeros([self.voc_size], dtype = self.dtype))
        # decoder output transform
        self.decoder_W = tf.Variable(tf.truncated_normal([self.decoder_units, self.voc_size ], 
                        stddev=1.0 / np.sqrt(self.encoder_units), dtype = self.dtype))
        self.decoder_b = tf.Variable(tf.zeros([self.voc_size], dtype = self.dtype))
        #wordEmbbding
        self.wordEmb_W = tf.Variable(tf.truncated_normal([self.voc_size, self.decoder_units // 2 ], 
                        stddev=1.0 / np.sqrt(self.encoder_units), dtype = self.dtype))
        self.wordEmb_b = tf.Variable(tf.zeros([self.decoder_units // 2 ], dtype = self.dtype))
        
        #output_embedding
        self.outputEmb_W = tf.Variable(tf.truncated_normal([self.encoder_units*2, self.decoder_units // 2 ], 
                        stddev=1.0 / np.sqrt(self.decoder_units // 2), dtype = self.dtype))
        self.outputEmb_b = tf.Variable(tf.zeros([self.decoder_units // 2 ], dtype = self.dtype))
        
        # connect all conponents (decoder, attention, decoder)
        self.encoder_output, encoder_final_state = self.Encoder()
        self.attention_output = self.Attention(self.encoder_output)
        self.decoder_output = self.Decoder(self.attention_output, encoder_final_state)
        
        # compute model loss
        decoder_output_flat = tf.reshape(self.decoder_output, [-1, self.decoder_units])
        decoder_output_transform_flat = tf.nn.xw_plus_b(decoder_output_flat, self.decoder_W, self.decoder_b)
        #decoder_logits_flat = tf.add(tf.matmul(decoder_output_transform_flat, self.emb_W), self.emb_b)
        decoder_logits = tf.reshape(decoder_output_transform_flat, (self.batch_size, self.max_len, self.voc_size))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_logits, labels=self.ys)
        self.decoder_loss = tf.reduce_mean(cross_entropy)
        # predict tensor
        self.prediction = tf.argmax(decoder_output_transform_flat, 1)
        
        # define train_op and initialize variable
        self.train_op = tf.train.AdamOptimizer().minimize(self.decoder_loss)
        self.sess.run(tf.global_variables_initializer())
    def Encoder(self):
        # a list that length is batch_size, every element refers to the time_steps of corresponding input
        encoder_input = tf.nn.embedding_lookup(self.wordEmb_W, tf.cast(self.xs, tf.int32))
        if self.mode == 'train':
            inputs_length = self.inputs_length
        elif self.mode == 'test':
            inputs_length = self.inputs_length_test
        multirnn_cell = MultiRNNCell([LSTMCell(self.encoder_units) for _ in range(self.encoder_lay_Num)],  state_is_tuple=True)  
        (fw_outputs, bw_outputs), (fw_final_state, bw_final_state) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=multirnn_cell, cell_bw=multirnn_cell, inputs=encoder_input,
                                            sequence_length=inputs_length, dtype=self.dtype))
        output = tf.concat((fw_outputs, bw_outputs), axis = 2)
        self.print_code = bw_outputs
        final_state = fw_final_state
        return output, final_state

    def Attention(self, encoder_output): 
        attention_states = tf.transpose(encoder_output, [1, 0, 2]) # transpose to time major
        print(attention_states.shape)
        if self.mode == 'train':
            inputs_length = self.inputs_length
        elif self.mode == 'test':
            inputs_length = self.inputs_length_test
        attention_output = []
        for i in range(inputs_length.shape[0]):
            attention_output.append(attention_states[inputs_length[i]-1,i,:])
        #attention_output = attention_states[6,:,:]
        attention_output = tf.nn.xw_plus_b(attention_output,self.outputEmb_W,self.outputEmb_b)
        return attention_output
    
    def Decoder(self, attention_output, encoder_final_state):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:#time = 0
                # initialization
                input = tf.concat((attention_output, attention_output), axis = 1)
                state = (multirnn_cell.zero_state(self.batch_size, tf.float32))
                emit_output = None
                loop_state = None
                elements_finished = False
            else:
                emit_output = cell_output
                if self.mode == 'test':
                    transformed_output = tf.nn.xw_plus_b(cell_output, self.decoder_W, self.decoder_b)#decoder_units to vac_size 
                    transformed_output = tf.argmax(transformed_output, 1)
                    transformed_output = tf.nn.xw_plus_b(tf.one_hot(transformed_output, self.voc_size,on_value=1.0, off_value=0.0,axis=-1), self.wordEmb_W, self.wordEmb_b)#vac_size to decoder_units//2  
                elif self.mode == 'train':
                    ys_onehot = tf.one_hot(self.ys[:,(time-1)], self.voc_size,on_value=1.0, off_value=0.0,axis=-1)
                    transformed_output = tf.nn.xw_plus_b(ys_onehot, self.wordEmb_W, self.wordEmb_b)
                #input = tf.concat([transformed_output, attention_output], axis = 1)
                input = tf.concat([transformed_output, attention_output], axis = 1)
                state = cell_state
                loop_state = None
            elements_finished = (time >= self.max_len)
            return (elements_finished, input, state, emit_output, loop_state)

        rnn_cell = LSTMCell(self.decoder_units)
        multirnn_cell = MultiRNNCell([rnn_cell]*self.decoder_lay_Num,  state_is_tuple=True)  
        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(multirnn_cell, loop_fn)
        outputs = tf.transpose(emit_ta.stack(), [1, 0, 2]) # transpose for putting batch dimension to first dimension
        return outputs
    def fit(self, xs, ys, batch_size, epoch):
        data_len = len(xs)
        # split data in to batches
        for ep in range(1):
            batch_offset = 0
            ep_loss = 0
            batch_run = 0
            while batch_offset < (data_len-batch_size):
                _, batch_loss = self.sess.run([self.train_op, self.decoder_loss], 
                                              feed_dict = {self.xs : xs[batch_offset:batch_offset + batch_size],
                                                           self.ys : ys[batch_offset:batch_offset + batch_size],
                                                           self.inputs_length : self.get_inputs_length(batch_size , xs[batch_offset:batch_offset + batch_size])})
                                                           
                batch_offset += batch_size
                ep_loss += batch_loss
                batch_run += 1
                print(batch_loss , (batch_offset/data_len)*100,'%','epoch: ',epoch)
                if batch_run%1000 == 0:
                    self.save()
            self.save()
            ep_loss /= batch_run
            print('epoch {}, loss: {}'.format(ep + 1, ep_loss))
            
    def predict(self, x):
        index_list , code = self.sess.run([self.prediction, self.print_code] ,  feed_dict = {self.xs : self.create_index(x),
                                                                                             self.inputs_length_test : self.get_inputs_length(1 , self.create_index(x))})
        ans = ''
        for i in range(self.max_len):
            if index_list[i] > 3:
                ans += self.dictionary[index_list[i]]
        return ans
    def get_inputs_length(self , batch_size , x):
        inputs_length = np.zeros((batch_size))
        for i in range(batch_size):
            inputs_length[i] = self.max_len 
            for ii in range(self.max_len):
                if int(x[i][ii]) == 2:
                    inputs_length[i] = int(ii+1)     
                    break                    
        return inputs_length
    def create_index(self , x):
        test_x = np.zeros((1,self.max_len))
        for i in range(self.max_len):
            if i < len(x):
                if x[i] in self.dictionary:
                    test_x[0][i] = self.dictionary.index(x[i])
                else:
                    test_x[0][i] = 3
            elif i == len(x):
                test_x[0][i] = 2
            else:
                test_x[0][i] = 0
        return test_x
    def save(self):
        model_file = os.getcwd() + '/model_file/model.ckpt'
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        saver = tf.train.Saver()
        saver.save(self.sess, model_file)
        return

    def restore(self , mode):
        if mode == 'test':
            model_file = os.getcwd() + '/model_file/model_test/model.ckpt'
            print('hi')
        elif mode == 'train':
            model_file = os.getcwd() + '/model_file/model.ckpt'
        if os.path.isdir(os.path.dirname(model_file)):
            saver = tf.train.Saver()
            saver.restore(self.sess, model_file)
        return