OBJ1 = \
	src/Dataset.o \
	src/Layer.o \
	src/MultilayerPerceptron.o \
	src/Neuron.o \
	src/Trainer.o \
	src/Infer.o \

CXXFLAGS = -I src/ -Wall -Wextra -O2

train.out: $(OBJ1) src/mainTrain.o
	c++ $^ -o $@

data.out: $(OBJ1) src/mainData.o
	c++ $^ -o $@

infer.out: $(OBJ1) src/mainInfer.o
	c++ $^ -o $@

%.o: %.cpp
	c++ $(CXXFLAGS) -c $< -o $@

init:
	python3 -m venv venv \
	source venv/bin/activate \
	pip install pandas matplotlib

clean:
	rm -f $(OBJ1) train.out 

.PHONY: clean 