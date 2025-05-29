#pragma once

#include "../MultilayerPerceptron.hpp"
#include <algorithm>

class Trainer
{
private:
    std::ofstream csvLoss;
	std::ofstream csvAccuracy;
	bool isCSV;

	double computeLoss(double label, const std::vector<double>& prediction);
    bool earlyStop(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &valData);
    bool initCSV();
    void buildCSV(int epoch, double loss, double accuracy);

public:
	~Trainer();
	/**
	 * Create a trainer for a multilayer perceptron
	 * csv: true for csv output, false for no csv output (csv are loss and accuracy per epoch)
	 */
	Trainer(bool csv);
	bool train(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &trainingData, std::vector<std::vector<double>> &valData, int epoch, double learningRate);
};
