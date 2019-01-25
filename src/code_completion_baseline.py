import tflearn
import numpy

class Code_Completion_Baseline:
    lenPre = 5
    lenSuff = 2
    max_hole_size = 3
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    def one_hot(self, string):
        vector = [0] * (len(self.string_to_number)+1)
        vector[self.string_to_number[string]] = 1
        return vector
    
    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict() 
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1
        
        # prepare x,y pairs
        xs = []
        ys = []
        for hole_size in range(1,self.max_hole_size+1):
            for token_list in token_lists:
                for idx in range(hole_size,len(token_list),3):      #take hole_size as start idx to avoid getting the same examples
                    if idx > 0:                    
                        pre = []
                        target = []
                        suff = []
                        for prefixInd in range(-self.lenPre, 0):
                            if idx+prefixInd < 0:
                                pre.append(numpy.zeros(len(self.string_to_number)+1, dtype=numpy.int8))
                            else:                        
                                pre.append(self.one_hot(self.token_to_string(token_list[idx + prefixInd])))                    
                        if idx+hole_size+1 >= len(token_list):  #discard all examples without complete target
                            break;                      
                        for targetInd in range(0, hole_size):
                            target = target + self.one_hot(self.token_to_string(token_list[idx + targetInd]))
                        for targetInd in range(hole_size,self.max_hole_size): #pad ys to same length with 'no-word'-vector
                            no_word_vec = numpy.zeros(len(self.string_to_number)+1, dtype=numpy.int8)
                            no_word_vec[len(self.string_to_number)] = 1                            
                            target = target + no_word_vec.tolist()
                        for suffixInd in range(hole_size, hole_size+self.lenSuff):
                            if idx+suffixInd < len(token_list):
                                suff.append(self.one_hot(self.token_to_string(token_list[idx + suffixInd])))
                            else:
                                suff.append(numpy.zeros(len(self.string_to_number)+1, dtype=numpy.int8)) 
                        xs.append(pre+suff)
                        ys.append(target)    
        print("x,y pairs: " + str(len(xs)))        
        
        for i in range(len(xs)):
            assert(len(xs[i]) == self.lenPre+self.lenSuff)
            assert(len(ys[i]) == (len(self.string_to_number)+1)*self.max_hole_size)
        return (xs, ys)

    def create_network(self):
        self.net = tflearn.input_data(shape=[None, self.lenPre+self.lenSuff, len(self.string_to_number)+1])   
        self.net = tflearn.simple_rnn(self.net, 128)
        self.net = tflearn.fully_connected(self.net, 128)        
        self.net = tflearn.fully_connected(self.net, 128)
        self.net = tflearn.fully_connected(self.net, (len(self.string_to_number)+1)*self.max_hole_size, activation='sigmoid')
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net)
    
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=50, batch_size=1024, show_metric=True)
        self.model.save(model_file)
        
    def query(self, prefix, suffix):
        pre = []
        suff = []
        for prefixInd in range(-self.lenPre, 0):
            if -prefixInd > len(prefix):
                pre.append(numpy.zeros(len(self.string_to_number)+1))
            else:                        
                pre.append(self.one_hot(self.token_to_string(prefix[prefixInd])))                    
        for suffixInd in range(0, self.lenSuff):
            if suffixInd < len(suffix):
                suff.append(self.one_hot(self.token_to_string(suffix[suffixInd])))
            else:
                suff.append(numpy.zeros(len(self.string_to_number)+1)) 
        assert(len(pre+suff) == self.lenPre+self.lenSuff)
        y = self.model.predict([pre+suff])
        predicted_seq = y[0]
        predicted_seq = numpy.reshape(predicted_seq,[self.max_hole_size, len(self.string_to_number)+1])
        
        res = []
        for i in range(0,self.max_hole_size):
            pred_tok = predicted_seq[i]
            pred_tok = pred_tok.tolist()
            best_number = pred_tok.index(max(pred_tok))
            if best_number != len(self.string_to_number):
                best_string = self.number_to_string[best_number]
                best_token = self.string_to_token(best_string)
                res.append(best_token)
            else:
                break
        return res
    
