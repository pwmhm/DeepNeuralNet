import numpy as np

# scribbles: to be deleted later
# data should have a structure of:
# [
#   [input, output],
#   [input, output],
#   ... 
# ]
# or
# [
#   [[input, output], [input, output], ...]],
#   [----]
# ]
# when using batches.
# 
# Dataloader's task is to process the data (shuffle, augment, etc) and provide
# the end user to costumize sampling strategies.
# it has to be able to iterate over the dataset.
# Data should be [input1, ..., ..., inputn],[output1]

class Dataloader():
    def __init__(self, Data, Batch):
        self.input = np.array(Data[0])
        self.gt = np.array(Data[1])

        self._length = len(self.input)//Batch
        self._index = 0

        # self.input = np.array_split(self.input, self._length)
        # self.gt = np.array_split(self.gt, self._length)

    def __next__(self):
        if self._index < self._length:
            retval = [self.input[self._index], self.gt[self._index]]
            self._index +=1
            return retval
        else:
            self._index = 0
            raise StopIteration

    def __iter__(self):
        return self
