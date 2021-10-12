#!/usr/bin/env bash

echo "Make project structure"
mkdir -p trained_models/{baseline,single,loop,logs}
mkdir -p trained_models/baseline/{densenet,vgg}
mkdir -p trained_models/single/{densenet,vgg}
mkdir -p trained_models/loop/{densenet,vgg}