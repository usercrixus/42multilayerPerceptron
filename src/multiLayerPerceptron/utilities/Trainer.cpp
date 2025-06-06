#include "Trainer.hpp"

Trainer::Trainer(bool csv) : isCSV(csv)
{
}

Trainer::~Trainer()
{
    if (csvAccuracyTrainData)
        csvAccuracyTrainData.close();
    if (csvLossTrainData)
        csvLossTrainData.close();
    if (csvAccuracyValidationData)
        csvAccuracyValidationData.close();
    if (csvLossValidationData)
        csvLossValidationData.close();
}

double Trainer::computeAcc(MultilayerPerceptron &mlp, const std::vector<std::vector<double>> &data)
{
    int correct = 0;
    for (const std::vector<double> &row : data)
    {
        std::vector<double> input(row.begin() + 1, row.end());
        std::vector<double> prediction = mlp.forward(input);
        int finalPrediction = (prediction[1] > prediction[0]) ? 1 : 0;
        if (finalPrediction == static_cast<int>(row[0]))
            ++correct;
    }
    return (static_cast<double>(correct) / data.size());
}

double Trainer::computeLoss(double label, const std::vector<double> &prediction)
{
    double p1 = std::max(std::min(prediction[1], 1.0 - 1e-15), 1e-15);
    if (label == 1.0)
        return -1 * std::log(p1);
    else
        return -1 * std::log(1.0 - p1);
}

bool Trainer::earlyStop(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &valData)
{
    static int noImprove = 0;
    static double bestVal = 0.0;
    double valAcc = computeAcc(mlp, valData);
    if (valAcc > bestVal)
    {
        bestVal = valAcc;
        noImprove = 0;
    }
    else if (++noImprove >= 10)
        return (std::cout << "Early stopping triggered" << std::endl, true);
    return (false);
}

bool Trainer::initCSV()
{
    if (isCSV)
    {
        csvLossTrainData = std::ofstream("csvLossTrainData.csv");
        csvAccuracyTrainData = std::ofstream("csvAccuracyTrainData.csv");
        csvLossValidationData = std::ofstream("csvLossValidationData.csv");
        csvAccuracyValidationData = std::ofstream("csvAccuracyValidationData.csv");
        if (!csvLossTrainData || !csvAccuracyTrainData || !csvLossValidationData || !csvAccuracyValidationData)
            return (false);
    }
    return (true);
}

void Trainer::buildCSV(int epoch, double lossTrain, double accuracyTrain, double lossVal, double accuracyVal)
{
    csvLossTrainData << epoch << "," << lossTrain << std::endl;
    csvAccuracyTrainData << epoch << "," << accuracyTrain << std::endl;
    csvLossValidationData << epoch << "," << lossVal << std::endl;
    csvAccuracyValidationData << epoch << "," << accuracyVal << std::endl;
}

bool Trainer::train(MultilayerPerceptron &mlp, std::vector<std::vector<double>> &trainingData, std::vector<std::vector<double>> &valData, int epoch, double learningRate)
{
    double trainLoss;
    double vasLoss;
    if (isCSV && !initCSV())
        return (std::cout << "Error during csv init" << std::endl, false);
    std::mt19937 gen{std::random_device{}()};
    int e = 0;
    while (e++ < epoch && !earlyStop(mlp, valData))
    {
        std::shuffle(trainingData.begin(), trainingData.end(), gen);
        trainLoss = 0.0;
        for (const std::vector<double> &row : trainingData)
        {
            std::vector<double> input{row.begin() + 1, row.end()};
            std::vector<double> prediction = mlp.forward(input);
            trainLoss += computeLoss(row[0], prediction);
            mlp.trainStep(input, row[0], learningRate);
        }
        vasLoss = 0.0;
        for (const std::vector<double> &row : valData)
        {
            std::vector<double> input{row.begin() + 1, row.end()};
            std::vector<double> prediction = mlp.forward(input);
            vasLoss += computeLoss(row[0], prediction);
        }
        buildCSV(e, trainLoss / trainingData.size(), computeAcc(mlp, trainingData), vasLoss / valData.size(), computeAcc(mlp, valData));
    }
    return (true);
}
