#include "Dataset.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
        return (std::cout << "Usage error. 3 args needed. Data csv path & separator & data object output path" << std::endl, 1);
    Dataset d(argv[1], argv[2][0]);
    if (!d.loadDatasetCSV())
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    d.normalize();
    d.shuffle();
    d.splitData(0.2);
    if (!d.saveDatasetObject(argv[3]))
        return (std::cout << "Error during the saving of " << argv[3] << std::endl, 1);
    return (0);
}
