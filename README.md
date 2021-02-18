# Continiual Few Shot Learning

## Requirements
- PyTorch 1.5.1+
    - Follow instructions here to set it up locally (depends on your environment)

## Getting Started
First, you need to setup the CLS module before using it with any of the available frameworks.

1. Change into the `cls_module` directory
2. Execute the `python setup.py develop` command to install the package and its dependencies

## Frameworks

### Omniglot Lake Benchmark
This is an implementation of the one-shot generalization benchmark introduced by Lake. The code is available under the
directory `frameworks/lake`.

To run an experiment using the Lake framework, you will need a valid configuration file. There is an existing configuration
file located in `frameworks/lake/definitions/aha_config.json` with the default configuration.

Run the experiment using `python oneshot_cls.py --config path/to/config.json`


