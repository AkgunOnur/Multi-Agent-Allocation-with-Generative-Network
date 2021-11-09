# from argparse import ArgumentTypeError
import numpy as np
# import os
import pickle

class Library():
    #Initialize library
    def __init__(self,library_size=50):
        self.library_size = library_size
        self.train_library = [[],[]]
        #Load test maps and add it to test_library
        test_libfile = open('./test_maps/test_map_library.pkl', 'rb')
        self.test_library = pickle.load(test_libfile)

    def add(self, map, label,opt):
        self.train_library[0].append(np.array(map).reshape((3,opt.full_map_size,opt.full_map_size)))
        self.train_library[1].append(np.array(label))
        print("Library size increased:", len(self.train_library[0]))
        self.save_maps(opt.mode)

    def get(self):
        rindex = np.random.randint(0,3)#len(self.train_library[0])
        return self.train_library[0][rindex], self.train_library[1][rindex]
    
    def save_maps(self,name):
        with open(f'./generated_maps/{name}/training_map_library.pkl', 'wb') as f:
            pickle.dump(self.train_library, f)