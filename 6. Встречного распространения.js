class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Initialize weights and biases with random values
    this.weightsInputHidden = this.randomMatrix(inputSize, hiddenSize);
    this.biasHidden = this.randomMatrix(1, hiddenSize);
    this.weightsHiddenOutput = this.randomMatrix(hiddenSize, outputSize);
    this.biasOutput = this.randomMatrix(1, outputSize);
  }

  randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() - 0.5),
    );
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  predict(input) {
    // Input to hidden layer
    const hidden = input.map(x =>
      this.sigmoid(x * this.weightsInputHidden[0][0] + this.biasHidden[0][0]),
    );

    // Hidden to output layer
    const output = hidden.map(x =>
      this.sigmoid(x * this.weightsHiddenOutput[0][0] + this.biasOutput[0][0]),
    );

    return output;
  }

  train(input, target, learningRate) {
    // Forward pass
    const hidden = input.map(x =>
      this.sigmoid(x * this.weightsInputHidden[0][0] + this.biasHidden[0][0]),
    );
    const output = hidden.map(x =>
      this.sigmoid(x * this.weightsHiddenOutput[0][0] + this.biasOutput[0][0]),
    );

    // Backpropagation
    // Backpropagation
    const outputError = target.map((t, i) => output[i] - t);
    const outputDelta = outputError.map((o, i) => o * output[i] * (1 - output[i]));

    const hiddenError = outputDelta.map((o, i) => o * this.weightsHiddenOutput[0][i]);
    const hiddenDelta = hiddenError.map((h, i) => h * hidden[i] * (1 - hidden[i]));

    // Update weights and biases
    this.weightsHiddenOutput[0] = this.weightsHiddenOutput[0].map(
      (w, i) => w - learningRate * outputDelta[i] * hidden[i],
    );
    this.biasOutput[0] = this.biasOutput[0].map((b, i) => b - learningRate * outputDelta[i]);
    this.weightsInputHidden[0] = this.weightsInputHidden[0].map(
      (w, i) => w - learningRate * hiddenDelta[i] * input,
    );
    this.biasHidden[0] = this.biasHidden[0].map((b, i) => b - learningRate * hiddenDelta[i]);
  }
}

// Example usage:

// Create a neural network with 1 input node, 3 hidden nodes, and 1 output node
const neuralNetwork = new NeuralNetwork(1, 3, 1);

// Train the neural network with some example data
for (let i = 0; i < 10000; i++) {
  const input = [Math.random()]; // Random input
  const target = [Math.sin(input[0])]; // Target value (sin function)
  neuralNetwork.train(input, target, 0.1);
}

// Test the trained neural network
const testInput = [0.5];
const predictedOutput = neuralNetwork.predict(testInput);
console.log(`Input: ${testInput[0]}, Predicted Output: ${predictedOutput[0]}`);
