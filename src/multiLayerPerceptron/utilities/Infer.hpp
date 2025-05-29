#pragma once

#include "../MultilayerPerceptron.hpp"
#include <iostream>

class Infer
{
private:
	MultilayerPerceptron &mlp;
	std::vector<std::vector<double>> &data;

public:
	Infer(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &data);
	~Infer();

    void getPredictions();
};
