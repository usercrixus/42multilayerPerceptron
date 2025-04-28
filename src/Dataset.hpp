#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>


class Dataset
{
private:
	std::vector<std::vector<double>> data;
	std::vector<std::vector<double>> validationData;
	std::vector<std::vector<double>> TrainingData;
	std::string fileName;
	char separator;
    bool isDatasetCorrect();

public:
	Dataset(std::string fileName, char separator);
    ~Dataset();

    bool loadDataset();
    void normalize();
    void shuffle();
    bool splitData(double validationPart);

	std::vector<std::vector<double>>& getTrainingData();
	std::vector<std::vector<double>>& getValidationData();
};
