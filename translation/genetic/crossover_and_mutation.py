
"""
number of ff/sa/dsa/ma
                    ----add ff/sa/dsa/ma
                    ----remove ff/sa/dsa/ma
properties of resnet/dense/pool
                    ----ff_dim
                    ----att_head

firstly, three basic operations:add, remove, alter
secondly, the particular operation is chosen based on a probability
"""
import random
import numpy as np
import copy

class CrossoverAndMutation(object):
    def __init__(self, individuals, args):
        self.prob_crossover = args.prob_crossover
        self.prob_mutation = args.prob_mutation
        self.individuals = individuals

    def process(self, args):
        crossover = Crossover(self.individuals, self.prob_crossover, args)
        offspring = crossover.do_crossover()
        self.offspring = offspring
        

        mutation = Mutation(self.offspring, self.prob_mutation, args)
        mutation.do_mutation()

        return offspring



class Crossover(object):
    def __init__(self, individuals, prob_, args):
        self.individuals = individuals
        self.prob = prob_

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = int(np.floor(np.random.random()*count_))
        idx2 = int(np.floor(np.random.random()*count_))
        while idx2 == idx1:
            idx2 = int(np.floor(np.random.random()*count_))

        if self.individuals[idx1].acc > self.individuals[idx1].acc:
            return idx1
        else:
            return idx2
    """
    binary tournament selection
    """
    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()

        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2
  

    def do_crossover(self):
        new_offspring_list = []
        for _ in range(len(self.individuals)//2):
            ind1, ind2 = self._choose_two_diff_parents()

            parent1, parent2 = copy.deepcopy(self.individuals[ind1]), copy.deepcopy(self.individuals[ind2])
            
            #encoder
            p_ = random.random()
            if p_ < self.prob:
                
                len1, len2 = len(parent1.encoder_units), len(parent2.encoder_units)
                pos1, pos2 = int(np.floor(np.random.random()*len1)), int(np.floor(np.random.random()*len2))
                assert pos1 < len1
                assert pos2 < len2

                unit_list1, unit_list2 = [], []
                for i in range(0, pos1):
                    unit_list1.append(parent1.encoder_units[i])
                for i in range(pos2, len(parent2.encoder_units)):
                    unit_list1.append(parent2.encoder_units[i])

                for i in range(0, pos2):
                    unit_list2.append(parent2.encoder_units[i])
                for i in range(pos1, len(parent1.encoder_units)):
                    unit_list2.append(parent1.encoder_units[i])

                # reorder the number of each unit based on its order in the list
                for i, unit in enumerate(unit_list1):
                    unit.number = i
                for i, unit in enumerate(unit_list2):
                    unit.number = i

                parent1.encoder_units = unit_list1
                parent2.encoder_units = unit_list2
                
            else:
                parent1.encoder_units = parent1.encoder_units
                parent2.encoder_units = parent2.encoder_units
                
            #decoder
            p_ = random.random()
            if p_ < self.prob:
                len1, len2 = len(parent1.decoder_units), len(parent2.decoder_units)
                pos1, pos2 = int(np.floor(np.random.random()*len1)), int(np.floor(np.random.random()*len2))
                assert pos1 < len1
                assert pos2 < len2

                unit_list1, unit_list2 = [], []
                for i in range(0, pos1):
                    unit_list1.append(parent1.decoder_units[i])
                for i in range(pos2, len(parent2.decoder_units)):
                    unit_list1.append(parent2.decoder_units[i])

                for i in range(0, pos2):
                    unit_list2.append(parent2.decoder_units[i])
                for i in range(pos1, len(parent1.decoder_units)):
                    unit_list2.append(parent1.decoder_units[i])
                        


                # reorder the number of each unit based on its order in the list
                for i, unit in enumerate(unit_list1):
                    unit.number = i
                for i, unit in enumerate(unit_list2):
                    unit.number = i

                parent1.decoder_units = unit_list1
                parent2.decoder_units = unit_list2
                
            else:
                parent1.decoder_units = parent1.decoder_units
                parent2.decoder_units = parent2.decoder_units

                
            offspring1, offspring2 = parent1, parent2
            offspring1.reset_acc()
            offspring2.reset_acc()
            new_offspring_list.append(offspring1)             
            offspring1, offspring2 = parent1, parent2
            offspring1.reset_acc()
            offspring2.reset_acc()
            new_offspring_list.append(offspring1)
            new_offspring_list.append(offspring2)
                
        return new_offspring_list


class Mutation(object):

    def __init__(self, individuals, prob_, args):
        self.individuals = individuals
        
        self.min_ff_dim = args.min_ff_dim
        self.max_ff_dim = args.max_ff_dim
        
        self.min_att_head = args.min_att_head
        self.max_att_head = args.max_att_head
        self.att_head = args.att_head
        self.mutation_list = args.mutation_list
        self.prob = prob_

    def do_mutation(self):
        mutation_list = self.mutation_list
        for indi in self.individuals:
            p_ = random.random()
            if p_ < self.prob:
                mutation_type = self.select_mutation_type(mutation_list)
                if mutation_type == 0:
                    self.do_add_unit_mutation(indi)
                    
                elif mutation_type == 1:
                    self.do_remove_unit_mutation(indi)
                    
                elif mutation_type == 2:
                    self.do_alter_mutation(indi)
                

    def do_add_unit_mutation(self, indi):
        """
        choose one position to add one unit, adding one resnet/densenet or pooling unit is determined by a probability of 1/3.
        However, if the maximal number of pooling units have been added into the current individual, only
        resnet/densenet unit will be add here
        """
        # determine the position where a unit would be added
        coder_type = random.randint(0, 1)
        if coder_type == 0:
            #encoder
            mutation_position = int(np.floor(np.random.random()*len(indi.encoder_units)))
            # determine the unit type for adding
            u_ = random.random()
            if u_ < 0.333:
                type_ = 1
            elif u_ < 0.666:
                type_ = 2
            else:
                type_ = 3

            if type_ == 2:
                add_unit = indi.init_a_encoder_sa(mutation_position+1, _att_head=None)

            elif type_ == 1:
                add_unit = indi.init_a_encoder_ff(mutation_position+1, _ff_dim=None)
            elif type_ == 3:
                add_unit = indi.init_a_encoder_dsa(mutation_position+1, _att_head=None)

            new_unit_list = []
            # add to the new list and update the number
            for i in range(mutation_position+1):
                new_unit_list.append(indi.encoder_units[i])
            new_unit_list.append(add_unit)
            for i in range(mutation_position+1, len(indi.encoder_units)):
                unit = indi.encoder_units[i]
                unit.number += 1
                new_unit_list.append(unit)
            indi.encoder_number_id += 1
            indi.encoder_units = new_unit_list
            indi.reset_acc()
            
        elif coder_type == 1:
            #decoder
            mutation_position = int(np.floor(np.random.random()*len(indi.decoder_units)))
            # determine the unit type for adding
            u_ = random.random()
            if u_ < 0.25:
                type_ = 1
            elif u_ < 0.50:
                type_ = 2
            elif u_ < 0.75:
                type_ = 3
            else:
                type_ = 4

            if type_ == 2:
                add_unit = indi.init_a_decoder_sa(mutation_position+1, _att_head=None)

            elif type_ == 1:
                add_unit = indi.init_a_decoder_ff(mutation_position+1, _ff_dim=None)
            elif type_ == 3:
                add_unit = indi.init_a_decoder_dsa(mutation_position+1, _att_head=None)
            elif type_ == 4:
                add_unit = indi.init_a_decoder_ma(mutation_position+1, _att_head=None)

            new_unit_list = []
            # add to the new list and update the number
            for i in range(mutation_position+1):
                new_unit_list.append(indi.decoder_units[i])
            new_unit_list.append(add_unit)
            for i in range(mutation_position+1, len(indi.decoder_units)):
                unit = indi.decoder_units[i]
                unit.number += 1
                new_unit_list.append(unit)
            indi.decoder_number_id += 1
            indi.decoder_units = new_unit_list
            indi.reset_acc()

    def do_remove_unit_mutation(self, indi):
        coder_type = random.randint(0, 1)
        
        #encoder
        if coder_type == 0:
            if len(indi.encoder_units) > 1:
                mutation_position = int(np.floor(np.random.random()*(len(indi.encoder_units)))) 
                new_unit_list = []
                for i in range(mutation_position):
                    new_unit_list.append(indi.encoder_units[i])
                for i in range(mutation_position+1, len(indi.encoder_units)):
                    unit = indi.encoder_units[i]
                    unit.number -= 1
                    new_unit_list.append(unit)
                indi.encoder_number_id -= 1
                indi.encoder_units = new_unit_list
                indi.reset_acc()
                
        #decoder       
        elif coder_type == 1:
            if len(indi.encoder_units) > 1:
                mutation_position = int(np.floor(np.random.random()*(len(indi.decoder_units)))) 
                new_unit_list = []
                for i in range(mutation_position):
                    new_unit_list.append(indi.decoder_units[i])
                for i in range(mutation_position+1, len(indi.decoder_units)):
                    unit = indi.decoder_units[i]
                    unit.number -= 1
                    new_unit_list.append(unit)
                indi.decoder_number_id -= 1
                indi.decoder_units = new_unit_list
                indi.reset_acc()

    def do_alter_mutation(self, indi):
        """
                ----ff_dim of ff
                ----att_head of sa, dsa, ma

        """
        coder_type = random.randint(0, 1)
        
        #encoder
        if coder_type == 0:
            mutation_position = int(np.floor(np.random.random()*len(indi.encoder_units)))
            mutation_unit = indi.encoder_units[mutation_position]

            if mutation_unit.type == 1:
                mutation_unit.ff_dim = np.random.randint(self.min_ff_dim, self.max_ff_dim+1)
                indi.encoder_units[mutation_position] = mutation_unit 
                indi.reset_acc()
                
            else:
                i = np.random.randint(0, len(self.att_head))
                att_head = self.att_head[i]
                
                mutation_unit.att_head = att_head
                indi.encoder_units[mutation_position] = mutation_unit 
                indi.reset_acc()               
        
        #decoder
        if coder_type == 1:
            mutation_position = int(np.floor(np.random.random()*len(indi.decoder_units)))
            mutation_unit = indi.decoder_units[mutation_position]

            if mutation_unit.type == 1:
                mutation_unit.ff_dim = np.random.randint(self.min_ff_dim, self.max_ff_dim+1)
                indi.decoder_units[mutation_position] = mutation_unit 
                indi.reset_acc()
                
            else:
                i = np.random.randint(0, len(self.att_head))
                att_head = self.att_head[i]
                
                mutation_unit.att_head = att_head
                indi.decoder_units[mutation_position] = mutation_unit 
                indi.reset_acc()               
              


    def select_mutation_type(self, _a):
        a = np.asarray(_a)
        k = 1
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ +=sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index[0]


