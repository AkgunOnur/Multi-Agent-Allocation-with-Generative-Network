import numpy as np
import os
import pickle
from generate_random_map import generate_random_map

class Library():
    #Initialize library
    def __init__(self,library_size=25):
        self.library_size = library_size
        self.library_maps = []

    def evaluate(self, map,reward):
        #If library is not full
        if(len(self.library_maps)<self.library_size):
            self.library_maps.append([map,reward])
        else:
            # library full find the best reward giving map and discard it
            # than add new map to the library
            self.library_maps.sort(reverse=False, key=lambda l: l[1])
            if(reward<=self.library_maps[-1][1]):
                self.library_maps.pop()
                self.library_maps.append([map,reward])
    
    def save_maps(self):
        #print("self.library_maps:", self.library_maps)
        with open('training_map_library.pkl', 'wb') as f:
            pickle.dump(self.library_maps, f)

# if __name__ == "__main__":
#     lib = Library(2)
#     for i in range(5):
#         map = generate_random_map(10)
#         lib.evaluate(map,np.random.randint(-10,10))
#     lib.save_maps()