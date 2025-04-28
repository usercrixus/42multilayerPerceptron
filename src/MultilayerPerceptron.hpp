#pragma once

#include <cmath>
#include <vector>
#include "Layer.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>

class MultilayerPerceptron
{
private:
	std::vector<Layer> layers;
	double learningRate;

public:
	MultilayerPerceptron();
	MultilayerPerceptron(std::vector<size_t> layerSizes);
	~MultilayerPerceptron();

	std::vector<double> forward(std::vector<double>& input);
    void trainStep(std::vector<double> &input, double trueLabel, double learningRate);
	bool saveModelObject(const std::string &filename);
    bool loadModelObject(const std::string &filename);
};
