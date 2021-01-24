import numpy as np
import hashlib
import copy

class Node(object):
    def __init__(self, number):
        self.number = number


class FFNode(Node):
    def __init__(self, number, ff_dim): #prob < 0.5
        super().__init__(number)
        self.type = 1
        self.ff_dim= ff_dim


class SANode(Node):
    def __init__(self, number, att_head):
        super().__init__(number)
        self.type = 2
        self.att_head = att_head


class DSANode(Node):
    def __init__(self, number, att_head):
        super().__init__(number)
        self.type = 3
        self.att_head = att_head

class MANode(Node):
    def __init__(self, number, att_head):
        super().__init__(number)
        self.type = 4
        self.att_head = att_head
        
class Individual(object):
    def __init__(self, args, indi_no):
        self.acc = -1.0
        self.id = indi_no # for record the id of current individual
        self.encoder_number_id = 0 # for record the total number of nodes 
        self.decoder_number_id = 0
        self.max_len = args.max_len
        self.max_encoder_len = args.max_encoder_len
        self.max_decoder_len = args.max_decoder_len
       
        self.min_encoder_ff = args.min_encoder_ff
        self.max_encoder_ff = args.max_encoder_ff
        self.min_encoder_sa = args.min_encoder_sa
        self.max_encoder_sa = args.max_encoder_sa
        self.min_encoder_dsa = args.min_encoder_dsa
        self.max_encoder_dsa = args.max_encoder_dsa
        
        self.min_decoder_ff = args.min_decoder_ff
        self.max_decoder_ff = args.max_decoder_ff
        self.min_decoder_sa = args.min_decoder_sa
        self.max_decoder_sa = args.max_decoder_sa
        self.min_decoder_dsa = args.min_decoder_dsa
        self.max_decoder_dsa =args.max_decoder_dsa
        self.min_decoder_ma = args.min_decoder_ma
        self.max_decoder_ma = args.max_decoder_ma
        
        self.min_ff_dim = args.min_ff_dim
        self.max_ff_dim = args.max_ff_dim
        
        self.min_att_head = args.min_att_head
        self.max_att_head = args.max_att_head
        self.att_head = args.att_head

                

        self.encoder_units = []
        self.decoder_units = []

    def reset_acc(self):
        self.acc = -1.0

    def initialize(self):
        # initialize how many resnet unit/pooling layer/densenet unit will be used
        num_ff = np.random.randint(self.min_encoder_ff , self.max_encoder_ff+1)
        num_sa = np.random.randint(self.min_encoder_sa , self.max_encoder_sa+1)
        num_dsa = np.random.randint(self.min_encoder_dsa, self.max_encoder_dsa+1)
        #num_ma = np.random.randint(self.min_ma, self.max_ma+1)

        # find the position where the pooling layer can be connected
        #encoder
        total_length = num_ff + num_sa + num_dsa
         
            
        all_positions = np.zeros(total_length, np.int32)
        if num_ff > 0: all_positions[0:num_ff] = 1;
        if num_sa > 0: all_positions[num_ff:num_ff+num_sa] = 2;
        if num_dsa > 0 : all_positions[num_ff+num_sa:num_ff+num_sa+num_dsa] = 3;
        for _ in range(total_length):
            np.random.shuffle(all_positions)
       

        # initialize the layers based on their positionsl
        for i in all_positions:
            if i == 1:
                ff = self.init_a_encoder_ff(_number=None, _ff_dim=None)
                self.encoder_units.append(ff)
            elif i == 2:
                sa = self.init_a_encoder_sa(_number=None, _att_head=None)
                self.encoder_units.append(sa)
            elif i == 3:
                dsa = self.init_a_encoder_dsa(_number=None, _att_head=None)
                self.encoder_units.append(dsa)
                
        #decoder       
        num_ff = np.random.randint(self.min_decoder_ff , self.max_decoder_ff+1)
        num_sa = np.random.randint(self.min_decoder_sa , self.max_decoder_sa+1)
        num_dsa = np.random.randint(self.min_decoder_dsa, self.max_decoder_dsa+1)
        num_ma = np.random.randint(self.min_decoder_ma, self.max_decoder_ma+1)

        # find the position where the pooling layer can be connected
        total_length = num_ff + num_sa + num_dsa + num_ma
        
            
        all_positions = np.zeros(total_length, np.int32)
        if num_ff > 0: all_positions[0:num_ff] = 1;
        if num_sa > 0: all_positions[num_ff:num_ff+num_sa] = 2;
        if num_dsa > 0 : all_positions[num_ff+num_sa:num_ff+num_sa+num_dsa] = 3;
        if num_ma > 0 : all_positions[num_ff+num_sa+num_dsa:num_ff+num_sa+num_dsa+num_ma] = 4;
        for _ in range(total_length):
            np.random.shuffle(all_positions)
            
        # initialize the layers based on their positionsl
        for i in all_positions:
            if i == 1:
                ff = self.init_a_decoder_ff(_number=None, _ff_dim=None)
                self.decoder_units.append(ff)
            elif i == 2:
                sa = self.init_a_decoder_sa(_number=None, _att_head=None)
                self.decoder_units.append(sa)
            elif i == 3:
                dsa = self.init_a_decoder_dsa(_number=None, _att_head=None)
                self.decoder_units.append(dsa)
            elif i == 4:
                ma = self.init_a_decoder_ma(_number=None, _att_head=None)
                self.decoder_units.append(ma)
                
    def re_initialize(self, encoder_seq, encoder_para, decoder_seq, decoder_para):
        index1 = 0
        index2 = 0
        
                      
        #encoder
        # initialize the layers based on their positionsl
        for i in encoder_seq:
            #ff
            if i == 1:
                ff = self.init_a_encoder_ff(_number=None, _ff_dim=encoder_para[index1])
                self.encoder_units.append(ff)
                
            #sa
            elif i == 2:
                sa = self.init_a_encoder_sa(_number=None, _att_head=encoder_para[index1])
                self.encoder_units.append(sa)
                
            #dsa
            elif i == 3:
                dsa = self.init_a_encoder_dsa(_number=None, _att_head=encoder_para[index1])
                self.encoder_units.append(dsa)
                
            index1 += 1
                
        #decoder       
        # initialize the layers based on their positionsl
        for i in decoder_seq:
            #ff
            if i == 1:
                ff = self.init_a_decoder_ff(_number=None, _ff_dim=decoder_para[index2])
                self.decoder_units.append(ff)
                
            #sa
            elif i == 2:
                sa = self.init_a_decoder_sa(_number=None, _att_head=decoder_para[index2])
                self.decoder_units.append(sa)
                
            #dsa
            elif i == 3:
                dsa = self.init_a_decoder_dsa(_number=None, _att_head=decoder_para[index2])
                self.decoder_units.append(dsa)
                
            #ma
            elif i == 4:
                ma = self.init_a_decoder_ma(_number=None, _att_head=decoder_para[index2])
                self.decoder_units.append(ma)
                
            index2 += 1
                   
           
    def init_a_encoder_ff(self, _number, _ff_dim):
        if _number:
            number = _number
        else:
            number = self.encoder_number_id
            self.encoder_number_id += 1
        if _ff_dim:
            ff_dim = _ff_dim
        else:
            ff_dim = np.random.randint(self.min_ff_dim, self.max_ff_dim+1)
       
        ff = FFNode(number, ff_dim)
        return ff
    
    def init_a_decoder_ff(self, _number, _ff_dim):
        if _number:
            number = _number
        else:
            number = self.decoder_number_id
            self.decoder_number_id += 1
            
        if _ff_dim:
            ff_dim = _ff_dim
            
        else:
            ff_dim = np.random.randint(self.min_ff_dim, self.max_ff_dim+1)
       
        ff = FFNode(number, ff_dim)
        return ff

    def init_a_encoder_sa(self, _number, _att_head):
        if _number:
            number = _number
        else:
            number = self.encoder_number_id
            self.encoder_number_id += 1

        if _att_head:
            att_head = _att_head
            
        else:
            i = np.random.randint(0, len(self.att_head))
            att_head = self.att_head[i]
            
                
        sa = SANode(number, att_head)
        return sa
    
    def init_a_decoder_sa(self, _number, _att_head):
        if _number:
            number = _number
        else:
            number = self.decoder_number_id
            self.decoder_number_id += 1

        if _att_head:
            att_head = _att_head
            
        else:
            i = np.random.randint(0, len(self.att_head))
            att_head = self.att_head[i]
            
        sa = SANode(number, att_head)
        return sa
    
    
    def init_a_encoder_dsa(self, _number, _att_head):
        if _number:
            number = _number
        else:
            number = self.encoder_number_id
            self.encoder_number_id += 1

        if _att_head:
            att_head = _att_head
            
        else:
            i = np.random.randint(0, len(self.att_head))
            att_head = self.att_head[i]
                        
        dsa = DSANode(number, att_head)
        return dsa
    
    def init_a_decoder_dsa(self, _number, _att_head):
        if _number:
            number = _number
        else:
            number = self.decoder_number_id
            self.decoder_number_id += 1

        if _att_head:
            att_head = _att_head
            
        else:
            i = np.random.randint(0, len(self.att_head))
            att_head = self.att_head[i]
            
        dsa = DSANode(number, att_head)
        return dsa

    def init_a_decoder_ma(self, _number, _att_head):
        if _number:
            number = _number
        else:
            number = self.decoder_number_id
            self.decoder_number_id += 1

        if _att_head:
            att_head = _att_head
            
        else:
            i = np.random.randint(0, len(self.att_head))
            att_head = self.att_head[i]
            
        ma = MANode(number, att_head)
        return ma


class Population(object):
    def __init__(self, args, gen_no):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        self.pop_size = args.pop_size
        self.individuals = []

    def initialize(self, args):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(args, indi_no)
            indi.initialize()
            self.individuals.append(indi)
            
    def re_initialize(self, args):
        en_para = [[4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2], [1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1], [4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1], [1, 2, 1, 1, 2, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 2, 4, 1], [2, 2, 1, 2, 2, 1, 2, 2, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1], [2, 2, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 2, 2, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1]]
        en_seq = [[2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1], [3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1], [2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1], [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1], [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1], [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1], [3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1], [2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1], [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1], [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1], [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1], [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1], [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1]]
        de_para = [[4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 2, 2, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 2, 2, 1], [1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1], [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [2, 2, 1, 2, 2, 1, 2, 2, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1], [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1], [2, 2, 1, 2, 2, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 2, 2, 1]]
        de_seq = [[2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], [1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1], [2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1]]
        
        index = 0
        for _ in range(self.pop_size):
            if index < len(en_seq):
                indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
                self.number_id += 1
                indi = Individual(args, indi_no)
                indi.re_initialize(en_seq[index], en_para[index], de_seq[index], de_para[index])
                self.individuals.append(indi)
                index += 1

            else:
                indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
                self.number_id += 1
                indi = Individual(args, indi_no)
                indi.initialize()
                self.individuals.append(indi)
                index += 1

            
    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            self.individuals.append(indi)


def test_individual(args):
    ind = Individual(args, 0)
    ind.initialize(args)
    print(ind)
    
def test_population(args):
    pop = Population(args, 0)
    pop.initialize()
    print(pop)



#if __name__ == '__main__':

    #test_individual(args)
    #test_population()






