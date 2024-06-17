# ZoMBI-Hop
 
![zombi](./figs/zombi-hop.png)


All the documentation regarding Line-BO is currently in the Line-BO repository: https://github.com/TadaSanar/Line-BO 

## Installation

To install, clone the following repository and sub-repository:

`$ git clone https://github.com/TadaSanar/ZoMBI-Hop-LineBO.git`

`$ cd ZoMBI-Hop-LineBO`

Install conda environment (this works only if you use conda) either with free package versions:

`$ conda env create -f environment20240617.yml`

Or use the environment file with fixed versions of packages:

`$ conda env create -f environment20240617fixedversions.yml`

(Optional: Install Spyder:

`$ conda install spyder`)

Run zombihop-line.py either by starting spyder and running the file, or by:

`$ python zombihop-line.py`

## Implementation

- The Line-BO functionality is implemented in linebo_fun.py and connected into ZoMBI-Hop via sampler.py.
- Linebo_XXX.py files are exactly the same versions than in the main repository ( https://github.com/TadaSanar/Line-BO ). In future, these py files will be removed from this repository and installed as links to the Line-BO repository (in other words: avoid modifying linebo_XXX.py and if you do, document the changes carefully).
- Sampler.py can be changed freely.

## TO DO

### Acquisition implementation considerations

ZoMBI-Hop uses an acquisition matrix by default. It is very slow with Line-BO, so I implemented acquisition function sampling as a function. This feature is many times faster than the matrix version. Both acquisition options can be run by modifying zombihop.py line 123:
- Acquisition matrix: `acq_type = None`
- Acquisition function (current implementation): `acq_type = self.acquisition_type`

### Zooming in with Line-BO

This is where I was left at in January. We need to decide whether to sample across the zoomed-in space or across the whole search space. I recommend this will be done by benchmarking the performance of the Line-BO ZoMBI-HOP with zoomed-in space vs. the whole search space.

The current implementation integrates the acquisition function only over the zoomed-in space. You can change the behavior by setting zombihop.py line 121 to:
`emin_global = None, emax_global = None,`

The challenge with using the zoomed-in space is that the start and end points A and B, respectively, of the selected line, will be provided from the edges of the zoomed-in space. And since ZoMBI-Hop zooms in a lot, the majority of the lines will be very short and the benefit of being able to synthesize many samples at one round diminishes.

On the other hand, the challenge with the whole search space is to confirm if the samples outside the zoomed-in subspace will be stored properly while ZoMBI-Hop runs; and if they will be actually participating enough to the optimization decisions _between_ the zoom-in phases.

### Constrained BO

For optimizing compositions, the sum of each input variable needs to sum up to 100%. This is currently not implemented in ZoMBI-Hop. Implementing the constraint(s) would increase the experiment applications for ZoMBI-Hop. See an example of how this is done in GPyOpt from linebo_wrappers.py/define_bo() line 61.

Line-BO has the constraint feature implemented via linebo_fun.py/choose_K() (set argument `constraint_sum_x = True`). The current implementation of LineBO-ZoMBI-Hop uses another function, sampler.py/choose_K_acq_zombihop(), which should be changed to linebo_fun.py/choose_K() (with argument `constraint_sum_x = True`) if you want to run constrained Line-BO with ZoMBI-Hop.

### General testing

There are also some other changes compared to the original ZoMBI-Hop. Ask Armi for details.

The code in general would benefit from testing and benchmarking since it has not been tested as well as the Line-BO with GPyOpt in the main repository ( https://github.com/TadaSanar/Line-BO ).

