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
    double getMaxOfColumn(int column);
    void normalize();
    void shuffle();
    bool splitData(double validationPart);

	const std::vector<std::vector<double>>& getTrainingData() const;
	const std::vector<std::vector<double>>& getValidationData() const;
};
