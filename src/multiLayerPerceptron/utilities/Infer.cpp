#include "Infer.hpp"

Infer::Infer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &data):
mlp(mlp),
data(data)
{
}

Infer::~Infer()
{
}

double Infer::computeLoss(double label, const std::vector<double> &prediction)
{
    double p1 = std::max(std::min(prediction[1], 1.0 - 1e-15), 1e-15);
    if (label == 1.0)
        return -1 * std::log(p1);
    else
        return -1 * std::log(1.0 - p1);
}


void Infer::getPredictions()
{
    for (auto& row : data) {
        std::vector<double> input(row.begin()+1, row.end());
        auto prediction = mlp.forward(input);
        std::cout << "Prediction loss: " << computeLoss(row[0], prediction) << std::endl;
    }
}
