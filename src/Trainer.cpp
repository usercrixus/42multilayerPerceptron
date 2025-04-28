#include "Trainer.hpp"

Trainer::Trainer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &valData, int epoch, double learningRate):
mlp(mlp),
data(data),
epoch(epoch),
learningRate(learningRate),
valData(valData)
{
}

Trainer::~Trainer()
{
}

double computeValidationAcc(MultilayerPerceptron &mlp, const std::vector<std::vector<double>> &valData)
{
    int correct = 0;
    for (auto& row : valData) {
        std::vector<double> input(row.begin()+1, row.end());
        auto prediction = mlp.forward(input);
        int finalPrediction = (prediction[1] > prediction[0]) ? 1 : 0;
        if (finalPrediction == static_cast<int>(row[0]))
            ++correct;
    }
    return (static_cast<double>(correct) / valData.size());
}

double Trainer::computeLoss(double label, const std::vector<double>& prediction)
{
    double p1 = std::max(std::min(prediction[1], 1.0 - 1e-15), 1e-15);
    return -(label * std::log(p1) + (1.0 - label) * std::log(1.0 - p1));
}
bool Trainer::earlyStop()
{
	static int noImprove = 0;
	static double bestVal = 0.0;
	double valAcc = computeValidationAcc(mlp, valData);
	if (valAcc > bestVal)
	{
		bestVal = valAcc;
		noImprove = 0;
	}
	else if (++noImprove >= 10)
		return (std::cout << "Early stopping triggered" << std::endl, true);
	return (false);
}

void Trainer::train()
{
    std::mt19937 gen{std::random_device{}()};
    double bestVal = 0.0;
    int noImprove = 0;
	int epoch = 0;
    while (epoch++ < this->epoch && !earlyStop())
    {
        std::shuffle(data.begin(), data.end(), gen);
        this->trainLoss = 0.0;
        for (const auto& row : data)
        {
            double label = row[0];
            std::vector<double> input{row.begin() + 1, row.end()};
            std::vector<double> prediction = mlp.forward(input);
            this->trainLoss += computeLoss(label, prediction);
            mlp.trainStep(input, label, learningRate);
        }
        double valAcc = computeValidationAcc(mlp, valData);
        std::cout << "Epoch " << epoch << "  trainLoss=" << trainLoss / data.size() << "  valAcc=" << valAcc << std::endl;
    }
}

