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
	std::vector<std::vector<double>> trainingData;
	std::string fileName;
	char separator;
    bool isDatasetCorrect();
    void saveVector(std::ofstream& out, const std::vector<std::vector<double>>& vector);
    void loadVector(std::ifstream &in, std::vector<std::vector<double>> &vector);

public:
	Dataset(std::string fileName, char separator);
    Dataset();
    ~Dataset();

    bool loadDatasetCSV();
    void normalize();
    void shuffle();
    bool splitData(double validationPart);

    bool saveDatasetObject(const std::string &filename);
    bool loadDatasetObject(const std::string &filename);

    std::vector<std::vector<double>>& getTrainingData();
	std::vector<std::vector<double>>& getValidationData();
};
