#include "Infer.hpp"

Infer::Infer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &data):
mlp(mlp),
data(data)
{
}

Infer::~Infer()
{
}

double Infer::computeValidationAcc()
{
    int correct = 0;
    for (auto& row : data) {
        std::vector<double> input(row.begin()+1, row.end());
        auto prediction = mlp.forward(input);
        int finalPrediction = (prediction[1] > prediction[0]) ? 1 : 0;
        if (finalPrediction == static_cast<int>(row[0]))
            ++correct;
    }
    return (static_cast<double>(correct) / data.size());
}
void Infer::getPredictions()
{
    double valAcc = computeValidationAcc();
    std::cout << "valAcc=" << valAcc << std::endl;
}
