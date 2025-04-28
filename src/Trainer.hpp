#pragma once

#include "MultilayerPerceptron.hpp"
#include <algorithm>

class Trainer
{
private:
	MultilayerPerceptron &mlp;
	std::vector<std::vector<double>> &data;
	std::vector<std::vector<double>> &valData;
	int epoch;
	double learningRate;
	double computeLoss(double label, const std::vector<double>& prediction);
    bool earlyStop();
    double trainLoss;

public:
	~Trainer();
	Trainer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &trainData, std::vector<std::vector<double>> &valData, int epochs, double lr);
	void train();
};

