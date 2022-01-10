#! /bin/bash
for d in $(ls seko_models); do
    python3 plotter.py --file_loc=${d}

done