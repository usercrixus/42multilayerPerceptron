OBJ1 = \
	src/Dataset.o \
	src/Layer.o \
	src/MultilayerPerceptron.o \
	src/Neuron.o \
	src/main.o \
	src/Trainer.o \
	src/Infer.o \

train.out: $(OBJ1)
	c++ $(OBJ1) -o $@

%.o: %.cpp
	c++ -c $< -o $@

clean:
	rm -f $(OBJ1) train.out 

.PHONY: clean 