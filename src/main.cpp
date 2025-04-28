#include "Dataset.hpp"
#include "MultilayerPerceptron.hpp"
#include <iostream>
#include "Trainer.hpp"
#include "Infer.hpp"

int main(int argc, char *argv[])
{
    Dataset d("data.csv", ',');
    if (!d.loadDataset())
        return 1;
    d.normalize();
    d.shuffle();
    d.splitData(0.2);

    //

    MultilayerPerceptron mlp({30, 24, 24, 2});
    Trainer train(mlp, d.getTrainingData(), d.getValidationData(), 1000, 0.01);
    train.train();

    //

    Infer infer(mlp, d.getValidationData());
    infer.getPredictions();
    return 0;
}
