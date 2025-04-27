#include <vector>
#include "Neuron.hpp"

class Layer {
private:
	std::vector<Neuron> neurons;
	size_t inputSize;
	size_t outputSize;
	double learningRate;

public:
	Layer(size_t inputSize, size_t output_size, double learningRate);
	~Layer();

    std::vector<double> forward(const std::vector<double> &input);
	void backward(const std::vector<double>& input, const std::vector<double>& deltas);

	std::vector<Neuron> getNeurons() const;
	size_t getInputSize() const;
};
