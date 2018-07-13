import gensim
import numpy



class preprocess():
    def __init__(self,modelpath):
        self.model_path = modelpath

    def load_models(self):
        self.model = gensim.models.Word2Vec.load(self.model_path)

    def fixedvec(self,data,seq_length):
        if (len(data) >= seq_length):
            data = data[:seq_length]
        else:
            miss = [[0] * 64 for _ in range(seq_length - len(data))]
            data.extend(miss)
        return numpy.array(data)

    def vector(self,v):
        try:
            return self.model[v]
        except:
            return [0]*64





















