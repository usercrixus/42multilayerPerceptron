#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class Neuron
{
private:
	std::vector<double> weights;
	double bias;
    double learningRate;

public:
	Neuron(size_t inputSize, double learningRate);
	~Neuron();

	double forward(const std::vector<double>& input);
    void backward(const std::vector<double> &input, double delta);

	std::vector<double> getWeights() const;
};
