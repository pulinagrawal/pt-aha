# AHA in PyTorch

## Requirements

- PyTorch >= 1.5.1 and < 1.9.0
  - Follow instructions here to set it up locally (depends on your environment)
- [cerenaut-pt-core](https://github.com/Cerenaut/cerenaut-pt-core) - Cerenaut's PyTorch core codebase

## Getting Started

First, you need to setup the CLS module before using it with any of the available frameworks.

1. Change into the `cls_module` directory
2. Execute the `python setup.py develop` command to install the package and its dependencies

## Frameworks

### Omniglot Lake Benchmark

This is an implementation of the one-shot generalization benchmark introduced by Lake. The code is available under the
directory `lake`.

Before you run the Lake benchmark, you will also have to install dependencies specific to this particular framework. All dependencies are listed inside `requirements.txt` inside the `lake/` directory.

You can install the dependencies using `pip install -r requirements.txt`.

To run an experiment using the Lake framework, you will need a valid configuration file. There is an existing configuration
file located in `lake/definitions/aha_config.json` with the default configuration.

Run the experiment using `python oneshot_cls.py --config path/to/config.json`

Using the config, you can select the type of Long Term Memory (LTM), currently the choice is between a simple k-sparse autoencoder or VGG.
The first stage in execution is to pre-train the LTM.
If you already have a trained model (in a checkpoint file), you can specify it in the config, and then skip the pre-training.
