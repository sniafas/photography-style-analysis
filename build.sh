#!/usr/bin/env bash

echo "Make project structure"
mkdir -p dataset/{csv,images}
mkdir -p trained_models/{baseline,randomly_results,single,loop,logs}
mkdir -p trained_models/baseline/{densenet,vgg}
mkdir -p trained_models/randomly_results/{densenet,vgg}
mkdir -p trained_models/single/{densenet,vgg}
mkdir -p trained_models/loop/{densenet,vgg}
