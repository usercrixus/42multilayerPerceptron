# Multilayer Perceptron C++ Framework

This project implements a Multilayer Perceptron (MLP) in C++, including modules for data loading, training, inference, and evaluation.  
Project Structure

```
dataInfer.csv  dataTrain.csv  en.subject.pdf  Makefile  README.md  src  venv

src:
dataset  mainData.cpp  mainData.o  mainInfer.cpp  mainInfer.o  mainTrain.cpp  mainTrain.o  multiLayerPerceptron  plotter

src/dataset:
Dataset.cpp  Dataset.hpp

src/multiLayerPerceptron:
Layer.cpp  Layer.hpp  MultilayerPerceptron.cpp  MultilayerPerceptron.hpp  Neuron.cpp  Neuron.hpp  utilities

src/multiLayerPerceptron/utilities:
Infer.cpp  Infer.hpp  Trainer.cpp  Trainer.hpp

src/plotter:
plotter.py
```

## Features

    Customizable MLP architecture (layers, neurons, activation functions).
    Simple dataset interface via Dataset.
    Supervised training using backpropagation (Trainer).
    Save/load model for inference purposes.

## ⚙️ Compilation

    make train.out
    make data.out
    make infer.out

## Running

Init

    python3 -m venv venv; source venv/bin/activate; pip install pandas matplotlib

Training

    ./train

Inference

    ./infer

Dataset Generation or Preprocessing

    ./data

### HELPER

Compile and execute:

    make data
    make train
    make infer


## Notes

    No external libraries required (100% pure C++).
    Compile with -O2 or -O3 for better performance.
    The framework can be extended with more activation functions, dropout, or other enhancements.
