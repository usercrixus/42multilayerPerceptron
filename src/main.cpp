#include "Dataset.hpp"
#include "MultilayerPerceptron.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
    Dataset d("data.csv", ',');
    if (!d.loadDataset())
        return 1;
    d.normalize();
    d.shuffle();
    d.splitData(0.1);

    // Create your MLP
    MultilayerPerceptron mlp({30, 24, 24, 1}, 0.05);

    // Training
    for (int epoch = 0; epoch < 1000; ++epoch) // 100 epochs
    {
        double totalLoss = 0.0;

        for (const auto& row : d.getTrainingData()) // <-- We need getter for TrainingData
        {
            std::vector<double> input(row.begin() + 1, row.end()); // features
            double label = row[0]; // true label

            double prediction = mlp.forward(input); // Predict
            double loss = mlp.binaryCrossEntropy(label, prediction); // Compute loss

            totalLoss += loss;
            mlp.trainStep(input, label); // Train
        }

        std::cout << "Epoch " << epoch << ", average loss = " << (totalLoss / d.getTrainingData().size()) << std::endl;

    }

    return 0;
}
