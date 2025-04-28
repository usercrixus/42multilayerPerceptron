#include "Dataset.hpp"
#include "MultilayerPerceptron.hpp"
#include <iostream>
#include "Trainer.hpp"
#include "Infer.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        return (std::cout << "Usage error. 2 args needed. dataset object path & model object path" << std::endl, 1);
	Dataset d;
	if (!d.loadDatasetObject(argv[1]))
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    MultilayerPerceptron mlp({30, 24, 24, 2});
    Trainer train(mlp, d.getTrainingData(), d.getValidationData(), 1000, 0.01);
    train.train();
    if (!mlp.saveModelObject(argv[2]))
        return (std::cout << "Error during the loading of " << argv[2] << std::endl, 1);
    return 0;
}
