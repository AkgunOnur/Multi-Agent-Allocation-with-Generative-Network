import numpy as np
import os
import pickle
from generate_random_map import generate_random_map

class Library():
    #Initialize library
    def __init__(self,library_size=180):
        self.library_size = library_size
        self.train_library = [[],[]]
        #Load test maps and add it to test_library
        test_libfile = open('./test_dataset.pickle', 'rb')
        self.test_library = pickle.load(test_libfile)
        #print("test lib[0]: ", np.asarray(self.test_library[0]).shape)
      

    def add(self, map, label):
        self.train_library[0].append(np.array(map).reshape((1,3,40,40)))
        self.train_library[1].append(np.array(label))
        print("Library size increased:", len(self.train_library[0]))
        #Save training library maps
        # print("map_shape: ", np.asarray(map).reshape((1,3,40,40)).shape)
        self.save_maps()
    
    def get(self):
        rindex = np.random.randint(0,len(self.train_library[0]))
        return self.train_library[0][rindex], self.train_library[1][rindex]
    
    def save_maps(self):
        #print("self.train_library:", self.train_library)
        with open('training_map_library.pkl', 'wb') as f:
            pickle.dump(self.train_library, f)

# if __name__ == "__main__":
#     l  = Library(1)
#     lib = Library(2)
#     for i in range(5):
#         map = generate_random_map(10)
#         lib.evaluate(map,np.random.randint(-10,10))
#     lib.save_maps()