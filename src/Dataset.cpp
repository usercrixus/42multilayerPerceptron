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
void Dataset::normalize()
{
    size_t featureCount = data[0].size() - 1; // Exclude label
    std::vector<double> means(featureCount, 0.0);
    std::vector<double> stddevs(featureCount, 0.0);

    // Compute means
    for (size_t col = 0; col < featureCount; ++col)
    {
        for (const auto& row : data)
            means[col] += row[col + 1];
        means[col] /= data.size();
    }
    // Compute standard deviations
    for (size_t col = 0; col < featureCount; ++col)
    {
        for (const auto& row : data)
            stddevs[col] += (row[col + 1] - means[col]) * (row[col + 1] - means[col]);
        stddevs[col] = std::sqrt(stddevs[col] / data.size());
    }
    // Normalize
    for (auto& row : data)
    {
        for (size_t col = 0; col < featureCount; ++col)
        {
            if (stddevs[col] != 0)
                row[col + 1] = (row[col + 1] - means[col]) / stddevs[col];
            else
                row[col + 1] = 0.0;
        }
    }
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
		validationData.push_back(data[i++]);
	while (i < data.size())
		TrainingData.push_back(data[i++]);
	return (true);
}

std::vector<std::vector<double>> &Dataset::getTrainingData() 
{
    return (TrainingData);
}

std::vector<std::vector<double>> &Dataset::getValidationData()
{
    return (validationData);
}