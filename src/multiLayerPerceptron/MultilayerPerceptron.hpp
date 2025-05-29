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
	/**
	 * Create a MultilayerPerceptron object
	 * layerSizes : exemple {30, 24, 24, 2}. The input size neurons.
	 * Here, 30 mean 30 data input, 24 mean 24 neuron input, then again 24 neuron input then 2 neuron output
	 */
	MultilayerPerceptron(std::vector<size_t> layerSizes);
	~MultilayerPerceptron();

	std::vector<double> forward(std::vector<double>& input);
    void trainStep(std::vector<double> &input, double trueLabel, double learningRate);
	bool saveModelObject(const std::string &filename);
    bool loadModelObject(const std::string &filename);
};
