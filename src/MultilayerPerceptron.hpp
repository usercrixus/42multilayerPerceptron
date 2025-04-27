#include <cmath>
#include <vector>
#include "Layer.hpp"
#include <iostream>

class MultilayerPerceptron
{
private:
	std::vector<Layer> layers;
	double learningRate;

public:
	MultilayerPerceptron(const std::vector<size_t>& layerSizes, double learningRate);
	~MultilayerPerceptron();

    double binaryCrossEntropy(double trueLabel, double predictedLabel);
	double forward(const std::vector<double>& input);
    void trainStep(const std::vector<double> &input, double trueLabel);
    std::vector<double> computeNewDelta(const Layer &layer, const std::vector<double> &nextDelta);
    double sigmoid(double x);
};
