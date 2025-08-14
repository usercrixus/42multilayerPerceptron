#include "Dataset.hpp"

Dataset::Dataset()
{
}

Dataset::~Dataset()
{
}

bool Dataset::loadDatasetCSV(std::string fileName)
{
    data.clear();
    std::ifstream file(fileName);
    if (!file.is_open())
        return (std::cerr << "Failed to open file: " << fileName << std::endl, false);

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

void Dataset::computeStats()
{
    size_t featureCount = data[0].size() - 1; // Exclude label
    featureMeans.assign(featureCount, 0.0);
    featureStddevs.assign(featureCount, 0.0);
    // Compute means
    for (size_t col = 0; col < featureCount; ++col)
    {
        for (const std::vector<double> &row : data)
            featureMeans[col] += row[col + 1];
        featureMeans[col] /= data.size();
    }
    // Compute standard deviations
    for (size_t col = 0; col < featureCount; ++col)
    {
        for (const std::vector<double> &row : data)
            featureStddevs[col] += (row[col + 1] - featureMeans[col]) * (row[col + 1] - featureMeans[col]);
        featureStddevs[col] = std::sqrt(featureStddevs[col] / data.size());
    }
}

void Dataset::normalize()
{
    size_t featureCount = data[0].size() - 1; // Exclude label
    // Normalize
    for (std::vector<double> &row : data)
    {
        for (size_t col = 0; col < featureCount; ++col)
        {
            if (featureStddevs[col] != 0)
                row[col + 1] = (row[col + 1] - featureMeans[col]) / featureStddevs[col];
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

void Dataset::saveVector(std::ofstream &out, const std::vector<std::vector<double>> &vector)
{
    size_t vectorSize = vector.size();
    out.write(reinterpret_cast<const char *>(&vectorSize), sizeof(vectorSize));
    for (const auto &row : vector)
    {
        size_t rowSize = row.size();
        out.write(reinterpret_cast<const char *>(&rowSize), sizeof(rowSize));
        out.write(reinterpret_cast<const char *>(row.data()), rowSize * sizeof(double));
    }
}

bool Dataset::saveDatasetObject(const std::string &filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return (false);
    }

    // 1) Save the 2D vectors (data, validationData, trainingData)
    saveVector(out, data);
    saveVector(out, validationData);
    saveVector(out, trainingData);

    // 2) Save featureMeans (1D)
    size_t meansCount = featureMeans.size();
    out.write(reinterpret_cast<const char *>(&meansCount), sizeof(meansCount));
    if (meansCount > 0)
    {
        out.write(reinterpret_cast<const char *>(featureMeans.data()), meansCount * sizeof(double));
    }

    // 3) Save featureStddevs (1D)
    size_t stddevCount = featureStddevs.size();
    out.write(reinterpret_cast<const char *>(&stddevCount), sizeof(stddevCount));
    if (stddevCount > 0)
    {
        out.write(reinterpret_cast<const char *>(featureStddevs.data()), stddevCount * sizeof(double));
    }

    out.close();
    return (true);
}

void Dataset::loadVector(std::ifstream &in, std::vector<std::vector<double>> &vector)
{
    size_t vectorSize;
    in.read(reinterpret_cast<char *>(&vectorSize), sizeof(vectorSize));
    vector.resize(vectorSize);
    for (auto &row : vector)
    {
        size_t rowSize;
        in.read(reinterpret_cast<char *>(&rowSize), sizeof(rowSize));
        row.resize(rowSize);
        in.read(reinterpret_cast<char *>(row.data()), rowSize * sizeof(double));
    }
}

bool Dataset::loadDatasetObject(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return (false);
    }
    // 1) Load the 2D vectors
    loadVector(in, data);
    loadVector(in, validationData);
    loadVector(in, trainingData);
    // 2) Load featureMeans (1D)
    size_t meansCount;
    in.read(reinterpret_cast<char *>(&meansCount), sizeof(meansCount));
    featureMeans.resize(meansCount);
    if (meansCount > 0)
    {
        in.read(reinterpret_cast<char *>(featureMeans.data()), meansCount * sizeof(double));
    }

    // 3) Load featureStddevs (1D)
    size_t stddevCount;
    in.read(reinterpret_cast<char *>(&stddevCount), sizeof(stddevCount));
    featureStddevs.resize(stddevCount);
    if (stddevCount > 0)
    {
        in.read(reinterpret_cast<char *>(featureStddevs.data()), stddevCount * sizeof(double));
    }

    in.close();
    return (true);
}

void Dataset::clear()
{
    data.clear();
    validationData.clear();
    trainingData.clear();
}

std::vector<std::vector<double>> &Dataset::getTrainingData()
{
    return (trainingData);
}

std::vector<std::vector<double>> &Dataset::getValidationData()
{
    return (validationData);
}

std::vector<std::vector<double>> &Dataset::getData()
{
    return (data);
}
