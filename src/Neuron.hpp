#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class Neuron
{
private:
	std::vector<double> weights;
	double bias;
	double lastSum;
	std::vector<double> lastInput;

public:
	Neuron(size_t inputSize);
	~Neuron();

	double forward(std::vector<double>& input);
    double backward(double delta, bool isHiddenLayer, double learningRate);
	
	std::vector<double> getWeights();
	double getLastSum();
	void setLastSum(double sum);
};
