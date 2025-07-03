# Bayesian Hyperparameter Optimization

This is a small demo project on using Bayesian Optimization for finding the best learning rate for a ResNet model on the Fashion MNIST dataset.

Supports the use of different epochs, batch sizes, and ResNet layer sizes. The optimizer assumes a logarithmic scaling unless logarithmic scaling is disabled. A Sobol sequence is used to generate initial samples for the Gaussian Process.


## Installation

Create a virtual environment (optional) and install requirements:

```bash
python -m venv .venv
. .venv/bin/activate  # assuming Linux/UNIX
pip install -r requirements.txt
```

If you are using Windows, run the `.venv\Scripts\activate` file. If you want to use CUDA on Windows you need to install PyTorch with CUDA manually. See [PyTorch's getting started](https://pytorch.org/get-started/locally/).


## How to use

There are two ways to use this demo: (i) run it as a Python script, and (ii) run it as a Jupyter Notebook. 


### (i) Python script

Basic usage would be to just run the script:

```bash
python optimize.py
```

This will run with default values, aka. 3 Sobol initialization steps, 7 Bayesian optimization steps, 256 training batch size, 10 training epochs, and a ResNet model with all layers of 16 hidden dimensions each. Figures will be saved under `./figures/`.

It can be further customized like this:

```bash
python optimize.py -i \  # i for interactive (display figure in blocking Window)
  --epochs 10 \  # train for 20 epochs each time
  --min-lr 1e-4 \  # use 0.0001 minimum LR
  --max-lr 1e-0  \  # use 1.0 maximum LR
  --batch-size 256 \  # use a batch size of 256
  --layers 16 32 64 128 256 512  # set the six ResNet layer sizes
```

Use `--debug` to test using a cheap objective function:

```bash
python optimize.py --debug
```

See the full usage below:

```text
usage: optimizer [-h] [--init-steps INIT_STEPS] [--optim-steps OPTIM_STEPS] [--min-lr MIN_LR] [--max-lr MAX_LR] [--no-log10] [--debug] [-i]
                 [-s SAVE_PATH] [--layers LAYERS LAYERS LAYERS LAYERS LAYERS LAYERS] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                 [--device DEVICE]

Bayesian Optimizer for finding the best learning rate for a ResNet model on the Fashion MNIST dataset.

options:
  -h, --help            show this help message and exit
  --init-steps INIT_STEPS
                        How many steps to initialize the bayesian optimizer with. (default: 3)
  --optim-steps OPTIM_STEPS
                        How many steps to perform Bayesian Optimization for. (default: 7)
  --min-lr MIN_LR       Minimum learning rate to check. (default: 0.0001)
  --max-lr MAX_LR       Maximum learning rate to check. (default: 1.0)
  --no-log10            Uses linear scaling instead of logarithmic. (default: False)
  --debug               Uses a much less expensive objective function for debug purposes. (default: False)
  -i, --interactive     Flag for displaying the figure interactively for each iteration in a blocking manner. (default: False)
  -s SAVE_PATH, --save-path SAVE_PATH
                        Where to save the figures, if specified. (default: ./figures/)
  --layers LAYERS LAYERS LAYERS LAYERS LAYERS LAYERS
                        The dimensionality of each of the six layers of the custom ResNet model. Used to customize the size of the model.
                        (default: [16, 16, 16, 16, 16, 16])
  --batch-size BATCH_SIZE
                        The batch size to use for training the ResNet model. (default: 256)
  --epochs EPOCHS       How many epochs to use for each training run of the ResNet model. (default: 5)
  --device DEVICE       Which device to run the training on, e.g. 'cuda' or 'cpu'. Option 'auto' chooses cuda if available, else CPU. With
                        multiple GPUs, specify the id with 'cuda:id', e.g.: 'cuda:0'. (default: auto)
```


### (ii) Jupyter Notebook

The [notebook.ipynb](notebook.ipynb)-file is provided for running it more interactively as a Notebook. Open it in a Jupyter server or with the [Jupyter VC code extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

The [test.ipynb](test.ipynb) contains some initial testing with Bayesian Optimization and a "minimal viable product", aka. it completes the task with just notebook code.
