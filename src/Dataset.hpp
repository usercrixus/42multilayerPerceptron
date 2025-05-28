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

    /**
     * load a csv file dataset
     * skip id value
     * set the diagnosis label (1 or 0)
     * load the other feature
     * return: true if all is right, false else.
     */
    bool loadDatasetCSV();
    /**
     * classic z-score standardization
     */
    void normalize();
    /**
     * shuffle the dataset to avoid bad distribution
     */
    void shuffle();
    /**
     * Split the dataset in 2 part, validationData and trainingData set.
     * return: true if all is right, false else. 
     */
    bool splitData(double validationPart);
    /**
     * Save the object (serialization)
     * filename: the output name of the object to save
     * return: true if all is right, false else.  
     */
    bool saveDatasetObject(const std::string &filename);
    /**
     * load the object (deserialization)
     * filename: the input name of the object to save
     * return: true if all is right, false else. 
     */
    bool loadDatasetObject(const std::string &filename);

    std::vector<std::vector<double>>& getTrainingData();
	std::vector<std::vector<double>>& getValidationData();
};
