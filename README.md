# Code for the article : "_Kernel Stein Discrepancy Descent_"



## Compat

This package has been developed and tested with `python3.7`. It is therefore not guaranteed to work with earlier versions of python.

## Install the repository on your machine


This package can easily be installed using `pip`, with the following command:

```bash
pip install -e .
```

This will install the package and all its dependencies, listed in `requirements.txt`.

## Reproducing the figures of the paper

To run the ICA experiment, run
```bash
python examples/ica.py
```

To display the effects of different initializations, run

```bash
python examples/plot_init.py
```

To display the effects of several different initializations with different mixtures of gaussians, run
```bash
python examples/plot_minima.py
```

To compare the running times of algorithms, run
```bash
python examples/plot_quantitative_expe_gaussian.py
```
where you can change the number of trials `n_expe` and the number of particles `n_samples`

To obtain the trajectory of different algorithms on a simple Gaussian density, run

```bash
python examples/plot_simple_gaussian.py
```

To see the behavior of the algorithm on skewed mixture of Gaussian run


```bash
python examples/plot_skewed.py
```

To see the effect of the annealing policy, run


```bash
python examples/plot_temperature.py
```
