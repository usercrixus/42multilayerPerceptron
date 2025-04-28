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

void Trainer::train()
{
	std::mt19937 gen{std::random_device{}()};
	double bestVal = 0.0;
	int noImprove = 0;
	for (int epoch = 0; epoch < this->epoch; ++epoch)
	{
		std::shuffle(data.begin(), data.end(), gen);
		double trainLoss = 0.0;
		for (const auto &row : data)
		{
			double label = row[0];
			std::vector<double> input{row.begin() + 1, row.end()};
			mlp.trainStep(input, label, learningRate);
		}
		double valAcc = computeValidationAcc(mlp, valData);
		std::cout << "Epoch " << epoch << "  trainLoss=" << trainLoss / data.size() << "  valAcc=" << valAcc << std::endl;
		if (valAcc > bestVal)
		{
			bestVal = valAcc;
			noImprove = 0;
		}
		else if (++noImprove >= 10)
		{
			std::cout << "Early stopping at epoch " << epoch << "\n";
			break;
		}
	}
}
