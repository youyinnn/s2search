import numpy as np, os, os.path as path

base_dir = path.join('./pipelining/exp6/scores')
files = os.listdir(base_dir)

for f in files: 
    if f.endswith('.npy'):
        arr = np.load(os.path.join(base_dir, f))
        np.savez_compressed(path.join(base_dir, f.replace('.npy', '.npz')), arr)
        os.remove(path.join(base_dir, f))