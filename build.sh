#!/usr/bin/env bash

echo "Make project structure"
mkdir -p dataset/{csv,images}
mkdir -p dataset/csv/active_learning
mkdir -p dataset/csv/active_learning/{all,all_ceal}
mkdir -p trained_models/{baseline,randomly_results,actively_results_sqal,actively_results_sqal_ceal,actively_results_all,actively_results_all_ceal,logs}
mkdir -p trained_models/randomly_results/{densenet,vgg}
mkdir -p trained_models/actively_results_sqal/{densenet,vgg}
mkdir -p trained_models/actively_results_sqal_ceal/{densenet,vgg}
mkdir -p trained_models/actively_results_all/{densenet,vgg}
mkdir -p trained_models/actively_results_all_ceal/{densenet,vgg}