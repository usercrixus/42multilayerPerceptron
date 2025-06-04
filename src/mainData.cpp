#include "dataset/Dataset.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
        return (std::cout << "Usage error. 2 args needed. data csv path (input) & data object path (output)" << std::endl, 1);
    Dataset d;
    if (!d.loadDatasetCSV(argv[1]))
        return (std::cout << "Error during the loading of " << argv[1] << std::endl, 1);
    d.computeStats();
    d.normalize();
    d.shuffle();
    d.splitData(0.2);
    if (!d.saveDatasetObject(argv[2]))
        return (std::cout << "Error during the saving of " << argv[3] << std::endl, 1);
    return (0);
}
