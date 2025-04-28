#include "Dataset.hpp"
#include "MultilayerPerceptron.hpp"
#include <iostream>
#include "Trainer.hpp"
#include "Infer.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        return (std::cout << "Usage error. 2 args needed. Dataset object path & model object path" << std::endl, 1);
	Dataset d;
	if (!d.loadDatasetObject(argv[1]))
        return (std::cout << "Error during the loading of "<< argv[1] << std::endl, 1);
    MultilayerPerceptron mlp;
    if (!mlp.loadModelObject(argv[2]))
        return (std::cout << "Error during the loading of " << argv[2] << std::endl, 1);
    Infer infer(mlp, d.getValidationData());
    infer.getPredictions();
    return 0;
}
