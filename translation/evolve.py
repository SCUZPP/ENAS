from fairseq import options
#import argument
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
import numpy as np
import copy
import pickle

class EvolveCNN(object):
    def __init__(self, args):
        self.pops = None

    def initialize_population(self, args):
        pops = Population(args, 0)
        pops.re_initialize(args)
        self.pops = pops

    def fitness_evaluate(self, args):
        fitness = FitnessEvaluate(self.pops.individuals, args)
        fitness.evaluate(args)


    def crossover_and_mutation(self, args):
        cm = CrossoverAndMutation(self.pops.individuals, args)
        offspring = cm.process(args)
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self, args):
        v_list = []
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc)


        #add log
        # find the largest one's index
        
        max_index = np.argmax(v_list)
        
        selection = Selection()
        selected_index_list = selection.RouletteSelection(v_list, k=args.pop_size)
        
        if max_index not in selected_index_list:
            first_selectd_v_list = [v_list[i] for i in selected_index_list]
            min_idx = np.argmin(first_selectd_v_list)
            selected_index_list[min_idx] = max_index

        next_individuals = [indi_list[i] for i in selected_index_list]

        next_gen_pops = Population(args, self.pops.gen_no+1)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops
    
    def do_work(self, args):
        gen_no = 0
        max_gen = args.max_gen
        self.initialize_population(args)
        self.fitness_evaluate(args)
        gen_no += 1
        
        save_path = 'population/population_gen_%s.pkl' % (self.pops.gen_no)
        pickle.dump(self.pops, open(save_path, 'wb'))   
        
        print('gen:',self.pops.gen_no)
        for ind in self.pops.individuals:
            print('ind.no', ind.id)
            print('ind.fit', ind.acc)
        
        for curr_gen in range(gen_no, max_gen):
         
            
            self.crossover_and_mutation(args)
            
            self.fitness_evaluate(args)
          
            self.environment_selection(args)
            
            save_path = 'population/population_gen_%s.pkl' % (self.pops.gen_no)
            pickle.dump(self.pops, open(save_path, 'wb'))   
            
            print('gen:',self.pops.gen_no)
            for ind in self.pops.individuals:
                print('ind.no', ind.id)
                print('ind.fit', ind.acc)


'''
args = argument.Args()
evoCNN = EvolveCNN(args)
evoCNN.do_work(args)
'''

          
if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    args.ddp_backend = 'no_c10d'
    
    '''
    args.seed = 1
    args.arch = 'man_iwslt_de_en'
    args.share_all_embeddings
    args.optimizer = 'adam'
    args.adam_betas = '(0.9, 0.98)'
    args.clip_norm = 0.0
    args.dropout = 0.3
    args.lr_scheduler = 'inverse_sqrt'
    args.warmup_init_lr = 1e-7 
    args.warmup_updates = 16000 
    args.lr = 7e-3 
    args.min_lr = 1e-09 
    args.criterion = 'label_smoothed_cross_entropy' 
    args.label_smoothing = 0.1 
    args.weight_decay  = 1e-4 
    args.max_tokens = 8192 
    args.save_dir = 'log/iwslt_de_en/man_iwslt_de_en_v'
    args.update_freq = 1 
    args.no_progress_bar_log_interval = 50 
    args.ddp_backend = 'no_c10d'
    args.save_interval_updates = 10000 
    args.keep_interval_updates = 5 
    args.max_epoch = 1 
    '''
    evoCNN = EvolveCNN(args)
    evoCNN.do_work(args)






