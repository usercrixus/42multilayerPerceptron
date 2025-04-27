#include "Dataset.hpp"


Dataset::Dataset(std::string fileName, char separator):
fileName(fileName),
separator(separator)
{
}

Dataset::~Dataset()
{
}

bool Dataset::loadDataset()
{
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << fileName << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line))
	{
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        // Skip ID value
		std::getline(ss, value, ',');
        // Read Diagnosis
        std::getline(ss, value, ',');
		if (value == "M")
			row.push_back(1);
		else
			row.push_back(0);
        // Read the others features
        while (std::getline(ss, value, ','))
            row.push_back(std::stod(value));
        data.push_back(row);
    }
    file.close();


    return (isDatasetCorrect());
}

double Dataset::getMaxOfColumn(int column)
{
    double max = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i][column] > max)
            max = data[i][column];
    }
    return max;
}

void Dataset::normalize()
{
    for (int col = 1; col < data[0].size(); ++col) // Start at 1 to skip label
    {
        double max = getMaxOfColumn(col);
        if (max == 0) continue; // avoid division by 0
        for (int row = 0; row < data.size(); ++row)
        {
            data[row][col] /= max;
        }
    }
}

bool Dataset::isDatasetCorrect()
{
	if (data.size() == 0)
		return (false);
	int len = data[0].size();
	for (const auto &d : data)
	{
		if (d.size() != len)
			return (std::cout << "Dataset is malformed" << std::endl, false);
	}
	return (true);
}

void Dataset::shuffle()
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}

bool Dataset::splitData(double validationPart)
{
	validationData.clear();
	TrainingData.clear();
	if (validationPart <= 0 || validationPart >= 1)
		return (std::cout << "Validation part should be > 0 and < 1" << std::endl, false);
	size_t i = 0;
	while (i < static_cast<size_t>(data.size() * validationPart))
	{
		validationData.push_back(data[i]);
		i++;
	}
	while (i < data.size())
	{
		TrainingData.push_back(data[i]);
		i++;
	}
	return (true);
}

const std::vector<std::vector<double>> &Dataset::getTrainingData() const
{
    return (TrainingData);
}

const std::vector<std::vector<double>> &Dataset::getValidationData() const
{
    return (validationData);
}