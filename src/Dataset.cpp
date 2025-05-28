#include "Dataset.hpp"


Dataset::Dataset(std::string fileName, char separator):
fileName(fileName),
separator(separator)
{
}

Dataset::Dataset()
{
}

Dataset::~Dataset()
{
}

bool Dataset::loadDatasetCSV()
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
	size_t len = data[0].size();
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
        for (const std::vector<double> &row : data)
            means[col] += row[col + 1];
        means[col] /= data.size();
    }
    // Compute standard deviations
    for (size_t col = 0; col < featureCount; ++col)
    {
        for (const std::vector<double> &row: data)
            stddevs[col] += (row[col + 1] - means[col]) * (row[col + 1] - means[col]);
        stddevs[col] = std::sqrt(stddevs[col] / data.size());
    }
    // Normalize
    for (std::vector<double> &row : data)
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
	trainingData.clear();
	if (validationPart <= 0 || validationPart >= 1)
		return (std::cout << "Validation part should be > 0 and < 1" << std::endl, false);
	size_t i = 0;
	while (i < static_cast<size_t>(data.size() * validationPart))
		validationData.push_back(data[i++]);
	while (i < data.size())
		trainingData.push_back(data[i++]);
	return (true);
}

void Dataset::saveVector(std::ofstream& out, const std::vector<std::vector<double>>& vector)
{
    size_t vectorSize = vector.size();
    out.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));
    for (const auto& row : vector) {
        size_t rowSize = row.size();
        out.write(reinterpret_cast<const char*>(&rowSize), sizeof(rowSize));
        out.write(reinterpret_cast<const char*>(row.data()), rowSize * sizeof(double));
    }
}

bool Dataset::saveDatasetObject(const std::string &filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        return (std::cerr << "Error opening file for writing: " << filename << std::endl, false);
    saveVector(out, data);
    saveVector(out, validationData);
    saveVector(out, trainingData);
    size_t fileNameSize = fileName.size();
    out.write(reinterpret_cast<const char*>(&fileNameSize), sizeof(fileNameSize));
    out.write(fileName.c_str(), fileNameSize);
    out.write(reinterpret_cast<const char*>(&separator), sizeof(separator));
    out.close();
    return (true);
}
void Dataset::loadVector(std::ifstream& in, std::vector<std::vector<double>>& vector)
{
    size_t vectorSize;
    in.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));
    vector.resize(vectorSize);
    for (auto& row : vector) {
        size_t rowSize;
        in.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
        row.resize(rowSize);
        in.read(reinterpret_cast<char*>(row.data()), rowSize * sizeof(double));
    }
}

bool Dataset::loadDatasetObject(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        return (std::cerr << "Error opening file for reading: " << filename << std::endl, false);
    loadVector(in, data);
    loadVector(in, validationData);
    loadVector(in, trainingData);
    size_t fileNameSize;
    in.read(reinterpret_cast<char*>(&fileNameSize), sizeof(fileNameSize));
    fileName.resize(fileNameSize);
    in.read(fileName.data(), fileNameSize);
    in.read(reinterpret_cast<char*>(&separator), sizeof(separator));
    in.close();
    return (true);
}

std::vector<std::vector<double>> &Dataset::getTrainingData() 
{
    return (trainingData);
}

std::vector<std::vector<double>> &Dataset::getValidationData()
{
    return (validationData);
}