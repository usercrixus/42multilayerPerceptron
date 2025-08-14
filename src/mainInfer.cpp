#include "dataset/Dataset.hpp"
#include "multiLayerPerceptron/MultilayerPerceptron.hpp"
#include <iostream>
#include "multiLayerPerceptron/utilities/Trainer.hpp"
#include "multiLayerPerceptron/utilities/Infer.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
        return (std::cout << "Usage error. 3 args needed. dataset object path (input) path & csvinfer path & model object path (input)" << std::endl, 1);
    Dataset d;
    if (!d.loadDatasetObject(argv[1]))
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    if (!d.loadDatasetCSV(argv[2]))
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    d.normalize();
    MultilayerPerceptron mlp;
    if (!mlp.loadModelObject(argv[3]))
        return (std::cout << "Error during the loading of " << argv[2] << std::endl, 1);
    Infer infer(mlp, d.getData());
    infer.getPredictions();
    return 0;
}
