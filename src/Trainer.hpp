#pragma once

#include "MultilayerPerceptron.hpp"
#include <algorithm>

class Trainer
{
private:
	MultilayerPerceptron &mlp;
	std::vector<std::vector<double>> &trainingData;
	std::vector<std::vector<double>> &valData;
	int epoch;
	double learningRate;
    double trainLoss;
    std::ofstream csvLoss;
	std::ofstream csvAccuracy;

	double computeLoss(double label, const std::vector<double>& prediction);
    bool earlyStop();
    void buildCSV(int epoch, double loss, double accuracy);

public:
	~Trainer();
	Trainer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &trainData, std::vector<std::vector<double>> &valData, int epochs, double lr, bool csv);
	void train();
};

