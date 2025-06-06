#pragma once

#include "../MultilayerPerceptron.hpp"
#include <algorithm>

class Trainer
{
private:
    std::ofstream csvLossTrainData;
	std::ofstream csvAccuracyTrainData;
    std::ofstream csvLossValidationData;
	std::ofstream csvAccuracyValidationData;
	bool isCSV;

	double computeLoss(double label, const std::vector<double>& prediction);
	double computeAcc(MultilayerPerceptron &mlp, const std::vector<std::vector<double>> &data);
    bool earlyStop(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &valData);
    bool initCSV();
    void buildCSV(int epoch, double lossTrain, double accuracyTrain, double lossVal, double accuracyVal);

public:
	~Trainer();
	/**
	 * Create a trainer for a multilayer perceptron
	 * csv: true for csv output, false for no csv output (csv are loss and accuracy per epoch)
	 */
	Trainer(bool csv);
    bool train(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &trainingData, std::vector<std::vector<double>> &valData, int epoch, double learningRate);
};
