#include "Infer.hpp"

Infer::Infer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &data):
mlp(mlp),
data(data)
{
}

Infer::~Infer()
{
}

void Infer::getPredictions()
{
    for (auto& row : data) {
        std::vector<double> input(row.begin()+1, row.end());
        auto prediction = mlp.forward(input);
        int finalPrediction = (prediction[1] > prediction[0]) ? 1 : 0;
        std::cout << "Prediction: " << finalPrediction << "(true one " << row[0] << ")" << std::endl;
    }
}
