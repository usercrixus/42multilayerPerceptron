#pragma once

#include <vector>
#include "Neuron.hpp"
#include <algorithm>

class Layer {
private:
	std::vector<Neuron> neurons;
	size_t inputSize;
	size_t outputSize;
	bool isHiddenLayer;

public:
	Layer(size_t inputSize, size_t output_size, bool hiddenLayer);
	~Layer();

    void activeNeuron(std::vector<double> &output);

    std::vector<double> forward(std::vector<double>& input);
	std::vector<double> backward(std::vector<double>& deltas, double learningRate);
    void ReLu(std::vector<double> &output);

	int getInputSize();
	int getOutputSize();
	bool getIsHiddenLayer();
	std::vector<Neuron> &getNeurons();

	void setNeurons(std::vector<Neuron> neurons);
};
