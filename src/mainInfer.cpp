#include "dataset/Dataset.hpp"
#include "multiLayerPerceptron/MultilayerPerceptron.hpp"
#include <iostream>
#include "multiLayerPerceptron/utilities/Trainer.hpp"
#include "multiLayerPerceptron/utilities/Infer.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
        return (std::cout << "Usage error. 3 args needed. Data csv & dataset.obj path & model object" << std::endl, 1);
    Dataset d;
    d.loadDatasetObject(argv[2]);
    if (!d.loadDatasetCSV(argv[1]))
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    d.normalize();
    MultilayerPerceptron mlp;
    if (!mlp.loadModelObject(argv[3]))
        return (std::cout << "Error during the loading of " << argv[3] << std::endl, 1);
    Infer infer(mlp, d.getData());
    infer.getPredictions();
    return 0;
}
