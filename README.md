## Overview
Most of the results can be seen using jupyter notebooks which run in your browser.

I have aggressively reorganized the code. An example using the new style is in `SimpleOptimization.ipynb`.
Whenever possible, future experiments should be conducted using this style.

Older files `SphericalProjectionExperiments.ipynb` and `vis_3D.py` are still useful and may be the best way to view aggregate statistics.

An initial toy example that we used at the beginning is presented in `ToyExample.ipynb`.
An example of actually optimizing based on the 2D plane but using a quadratic time to alarm function is shown in `QuadraticExample.ipynb`.
The experiments with real data are shown in `Experiements.ipynb`.
More recently, I've done experiments in `SphericalProjectionExperiments.ipynb` with the new data and spherical projections.
A more through analysis of the optimizer's performance is shown in ``

3D data which can't be visualized well in a notebook is shown in `vis_3D.py`.

## Setup
When you clone do `git clone --recurse-submodules https://github.com/russelldj/SmokeDetectorOptimization.git` This will include the code which does all the optimization.
The you will need to `cd detector-placement-module` to enter the directory and execute `python setup.py develop` to make the library accessible.

Python 3 is required. You will also need to install a variety of packages. The hopefully-current dependencies are in
`requirements.txt`. To install from that file do `pip install -r requirements.txt`

## Useful tricks
To use the python kernel in your current environment in jupyter, you need to explicitly set it up.
With the desired environment activated, run the following command, replacing `<yourname>` with a name represening this kernel..
`python -m ipykernel install --user --name placement-module --display-name <yourname>`
For example, setting your name to `smoke_detector`:
`python -m ipykernel install --user --name placement-module --display-name smoke_detector`
This should then appear in your list of kernels.

Some constants are stored in `detector-placement-module/src/smokedetectoroptimization/constants.py`.

In the jupyter notebook menu you can do select `File` -> `Download As` -> `Python (.py)` to get the file as a normal python script.
Running this on the command line will let you manipulate 3D plots.

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
