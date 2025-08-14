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
    std::vector<double> featureMeans;
    std::vector<double> featureStddevs;
    bool isDatasetCorrect();
    void saveVector(std::ofstream &out, const std::vector<std::vector<double>> &vector);
    void loadVector(std::ifstream &in, std::vector<std::vector<double>> &vector);

public:
    Dataset();
    ~Dataset();

    /**
     * load a csv file dataset
     * skip id value
     * set the diagnosis label (1 or 0)
     * load the other feature
     * return: true if all is right, false else.
     */
    bool loadDatasetCSV(std::string fileName);
    /**
     * cumpute stat to normalize
     */
    void computeStats(); 
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
    /**
     * Clear
     *      std::vector<std::vector<double>> data;
     *      std::vector<std::vector<double>> validationData;
     *      std::vector<std::vector<double>> trainingData;
     * 
     * You should use it after training, then save the new dataset
     * object (potentialy on the same name to erase the previous one)
     * then, reuse it the the infer part for normalization.
     */
    void clear();

    std::vector<std::vector<double>> &getTrainingData();
    std::vector<std::vector<double>> &getValidationData();
    std::vector<std::vector<double>> &getData();
    const std::vector<double> &getFeatureMeans() const { return featureMeans; }
    const std::vector<double> &getFeatureStddevs() const { return featureStddevs; }
};
