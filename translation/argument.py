class Args:
    def __init__(self):
        self.min_encoder_ff = 0
        self.max_encoder_ff = 3
        self.min_encoder_sa = 0
        self.max_encoder_sa = 3
        self.min_encoder_dsa = 0
        self.max_encoder_dsa = 3

        self.min_decoder_ff = 0
        self.max_decoder_ff = 3
        self.min_decoder_sa = 0
        self.max_decoder_sa = 3
        self.min_decoder_dsa = 0
        self.max_decoder_dsa = 3
        self.min_decoder_ma = 0
        self.max_decoder_ma = 3

        self.min_ff_dim = 0
        self.max_ff_dim = 8
        self.min_att_head = 0
        self.max_att_head = 8
        self.pop_size = 10
        self.prob_crossover = 0.1
        self.prob_mutation = 0.1
        self.max_gen = 4
        self.max_len = 21
        self.mutation_list = [0.4, 0.3, 0.3]
        self.att_head = [2,4,8,16]
        
        
    
