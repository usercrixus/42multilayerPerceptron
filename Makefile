OBJ1 = \
	src/dataset/Dataset.o \
	src/multiLayerPerceptron/Layer.o \
	src/multiLayerPerceptron/MultilayerPerceptron.o \
	src/multiLayerPerceptron/Neuron.o \
	src/multiLayerPerceptron/utilities/Trainer.o \
	src/multiLayerPerceptron/utilities/Infer.o \

CXXFLAGS = -I src/ -Wall -Wextra -Werror -std=c++17

train.out: $(OBJ1) src/mainTrain.o
	c++ $^ -o $@

data.out: $(OBJ1) src/mainData.o
	c++ $^ -o $@

infer.out: $(OBJ1) src/mainInfer.o
	c++ $^ -o $@

%.o: %.cpp
	c++ $(CXXFLAGS) -c $< -o $@

init:
	python3 -m venv venv; source venv/bin/activate;	pip install pandas matplotlib

clean:
	rm -f $(OBJ1)

fclean: clean
	rm -f train.out *.csv *.png *.out *.obj

.PHONY: clean fclean init
