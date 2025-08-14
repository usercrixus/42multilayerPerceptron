#include "dataset/Dataset.hpp"
#include "multiLayerPerceptron/MultilayerPerceptron.hpp"
#include <iostream>
#include "multiLayerPerceptron/utilities/Trainer.hpp"
#include "multiLayerPerceptron/utilities/Infer.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        return (std::cout << "Usage error. 2 args needed. dataset object path (input) & model object path (output)" << std::endl, 1);
    Dataset d;
    if (!d.loadDatasetObject(argv[1]))
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    MultilayerPerceptron mlp({30, 24, 24, 2});
    Trainer train(true);
    train.train(mlp, d.getTrainingData(), d.getValidationData(), 1000, 0.01);
    if (!mlp.saveModelObject(argv[2]))
        return (std::cout << "Error during the loading of " << argv[2] << std::endl, 1);
    d.clear();
    if (!d.saveDatasetObject(argv[1]))
        return (std::cout << "Error during the saving of " << argv[3] << std::endl, 1);

    if (system("python3 src/plotter/plotter.py csvLossTrainData.csv") < 0)
        return (std::cout << "Error during the display of the graph" << std::endl, 1);
    if (system("python3 src/plotter/plotter.py csvAccuracyTrainData.csv") < 0)
        return (std::cout << "Error during the display of the graph" << std::endl, 1);
    if (system("python3 src/plotter/plotter.py csvLossValidationData.csv") < 0)
        return (std::cout << "Error during the display of the graph" << std::endl, 1);
    if (system("python3 src/plotter/plotter.py csvAccuracyValidationData.csv") < 0)
        return (std::cout << "Error during the display of the graph" << std::endl, 1);
    return 0;
}
