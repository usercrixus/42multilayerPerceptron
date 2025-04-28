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

public:
	~Trainer();
	Trainer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &trainData, std::vector<std::vector<double>> &valData, int epochs, double lr);
	void train();
};

