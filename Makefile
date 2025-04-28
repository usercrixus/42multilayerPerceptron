OBJ1 = \
	src/Dataset.o \
	src/Layer.o \
	src/MultilayerPerceptron.o \
	src/Neuron.o \
	src/Trainer.o \
	src/Infer.o \

train.out: $(OBJ1) src/mainTrain.o
	c++ $^ -o $@

data.out: $(OBJ1) src/mainData.o
	c++ $^ -o $@

infer.out: $(OBJ1) src/mainInfer.o
	c++ $^ -o $@

%.o: %.cpp
	c++ -c $< -o $@

clean:
	rm -f $(OBJ1) train.out 

.PHONY: clean 