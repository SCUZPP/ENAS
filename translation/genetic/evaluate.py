import os
import sys
sys.path.append('..')
import train
import generate
import numpy as np
import torch
import pickle

class FitnessEvaluate(object):

    def __init__(self, individuals, args):
        self.individuals = individuals
        #p = pickle.load(open('score.pkl', 'rb'))
        #self.p = p


    def evaluate(self, args):
        
        for indi in self.individuals:
            flag = 0
            s = 0
            p = pickle.load(open('score.pkl', 'rb'))
            self.p = p
            #print('p', self.p)
            
            encoder_seq = []
            encoder_para = []

            decoder_seq = []
            decoder_para = []
            #encoder
            for unit in indi.encoder_units:
                if unit.type == 1:
                    encoder_para.append(unit.ff_dim)
                    
                else:
                    encoder_para.append(unit.att_head)
                    
                encoder_seq.append(unit.type)
                
            #decoder
            for unit in indi.decoder_units:
                if unit.type == 1:
                    decoder_para.append(unit.ff_dim)
                    
                else:
                    decoder_para.append(unit.att_head)
                    
                decoder_seq.append(unit.type)
                
            print('encoder_seq', encoder_seq)
            print('encoder_para', encoder_para)
            print('decoder_seq', decoder_seq)
            print('decoder_para', decoder_para)
            
            temp = encoder_seq + encoder_para + decoder_seq + decoder_para
            #print('temp',temp)
            for i in range (len(self.p)):
                #print('self.p',self.p[i])
                if temp == self.p[i]:
                    indi.acc = self.p[i + 1]
                    print('indi.acc', indi.acc)
                    flag = 1
                    break
                    

            if flag == 1:
                s = 1 
                
            else:
                          


                try: 
                    train.train_model(args, encoder_seq, encoder_para, decoder_seq, decoder_para)

                except RuntimeError:

                    indi.acc = 0
                    self.p.append(temp)
                    self.p.append(indi.acc)
                    save_path = 'score.pkl' 
                    pickle.dump(self.p, open(save_path, 'wb'))   

                    break

                try: 
                    score = generate.model_test(args, encoder_seq, encoder_para, decoder_seq, decoder_para)

                except RuntimeError:
                    indi.acc = 0
                    self.p.append(temp)
                    self.p.append(indi.acc)
                    save_path = 'score.pkl' 
                    pickle.dump(self.p, open(save_path, 'wb'))         
                    break

                torch.cuda.empty_cache()
                indi.acc = score 

                self.p.append(temp)
                self.p.append(indi.acc)
                save_path = 'score.pkl'
                pickle.dump(self.p, open(save_path, 'wb'))   

                #indi.acc = np.random.randint(0,100)


       
 







