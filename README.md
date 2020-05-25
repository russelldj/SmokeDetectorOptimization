## Overview

A good overview is provided in `StreamlinedExperiements.ipynb`. This is a jupyter notebook which you can run, but you can also view it in browser.
All the algorithms are in `SDOptimizer.py`

## Setup
When you clone do `git clone --recurse-submodules https://github.com/russelldj/SmokeDetectorOptimization.git` This will include the Platypus multiobjective optimizer library.
The you will need to `cd Platypus` to enter the directory and execute `python setup.py develop` to make the library accessible.

You will also need to install a variety of packages. Chief among them are `scipy`, `matplotlib`, `pandas`, and likely some others. They can be installed with the command `pip install <library>` or, if conda is installed `conda install <library>`.
`tqdm` is easiest to install using `pip install tqdm`

## Outputs
Most scripts which generate visualization will also write a camera-ready version of the visualization to the `./vis/` folder. Generally, the only difference is this figure does not include a title. Within a jupyter notebook, images can be right clicked and saved. In a matplotlib pop-up, there is an icon which allows you to save files.  

## TODO

- create a requirements.txt file listing all of the required libraries
