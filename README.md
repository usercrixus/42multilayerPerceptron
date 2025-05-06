📚 Multilayer Perceptron C++ Framework

This project implements a Multilayer Perceptron (MLP) in C++, including modules for data loading, training, inference, and evaluation.
📁 Project Structure

    src/
    ├── Dataset.*                # Data loading and handling
    ├── Infer.*                  # Inference using a trained model
    ├── Layer.*                  # Implementation of a neural network layer
    ├── MultilayerPerceptron.*   # Definition of the full MLP model
    ├── Neuron.*                 # Individual neuron logic
    ├── Trainer.*                # Supervised training logic
    ├── mainTrain.cpp            # Entry point for training
    ├── mainInfer.cpp            # Entry point for inference
    ├── mainData.cpp             # Dataset generation or preprocessing

🧠 Features

    Customizable MLP architecture (layers, neurons, activation functions).
    Simple dataset interface via Dataset.
    Supervised training using backpropagation (Trainer).
    Save/load model for inference purposes.

⚙️ Compilation

    make train.out
    make data.out
    make infer.out

🚀 Running
Training

    ./train
Inference

    ./infer
Dataset Generation or Preprocessing

    ./data

📌 Notes

    No external libraries required (100% pure C++).
    Compile with -O2 or -O3 for better performance.
    The framework can be extended with more activation functions, dropout, or other enhancements.
