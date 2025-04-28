#pragma once

#include <cmath>
#include <vector>
#include "Layer.hpp"
#include <iostream>
#include <algorithm>

class MultilayerPerceptron
{
private:
	std::vector<Layer> layers;
	double learningRate;

public:
	MultilayerPerceptron(std::vector<size_t> layerSizes);
	~MultilayerPerceptron();

	std::vector<double> forward(std::vector<double>& input);
    void trainStep(std::vector<double> &input, double trueLabel, double learningRate);
};
