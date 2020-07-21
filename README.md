## Overview
Most of the results can be seen using jupyter notebooks which run in your browser.


Currently the most useful files are `SphericalProjectionExperiments.ipynb` and `vis_3D.py`. An example using the new style is in `SimpleOptimization.ipynb`

An initial toy example that we used at the beginning is presented in `ToyExample.ipynb`.
An example of actually optimizing based on the 2D plane but using a quadratic time to alarm function is shown in `QuadraticExample.ipynb`.
The experiments with real data are shown in `Experiements.ipynb`.
More recently, I've done experiments in `SphericalProjectionExperiments.ipynb` with the new data and spherical projections.
A more through analysis of the optimizer's performance is shown in ``

3D data which can't be visualized well in a notebook is shown in `vis_3D.py`.

All the algorithms are in `SDOptimizer.py`

## Setup
When you clone do `git clone --recurse-submodules https://github.com/russelldj/SmokeDetectorOptimization.git` This will include the code which does all the optimization.
The you will need to `cd detector-placement-module` to enter the directory and execute `python setup.py develop` to make the library accessible.

You will also need to install a variety of packages. Chief among them are `jupyter`, `scipy`, `matplotlib`, `pandas`, and likely some others. They can be installed with the command `pip install <library>` or, if conda is installed `conda install <library>`.
`tqdm` is easiest to install using `pip install tqdm`

In the future I will provide a requirements.txt file which helps install all the dependencies.

## Useful tricks
Some constants are stored in `SDOptimizer/constants.py`.

## Outputs
Most scripts which generate visualization will also write a camera-ready version of the visualization to the `./vis/` folder. Generally, the only difference is this figure does not include a title. Within a jupyter notebook, images can be right clicked and saved. In a matplotlib pop-up, there is an icon which allows you to save files.  

## TODO
- create a requirements.txt file listing all of the required libraries
- create a 3D visualization of the data
- Figure out exactly what the Pareto optimal set is
- Think about if there's a good way to represent this as a graph
- Visualize the selected location in 3D

## Concerns
- I think it's doing a lookup on the interpolated grid. It should really do it on the original points

## Questions
- Why do so many locations alarm on the first timestep?
