const randomMatrix = (rows, cols) => {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix[i] = [];
    for (let j = 0; j < cols; j++) {
      matrix[i][j] = Math.random();
    }
  }
  return matrix;
};

const euclideanDistance = (x, y) => {
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    sum += Math.pow(x[i] - y[i], 2);
  }
  return Math.sqrt(sum);
};

const matrixMultiply = (matrix1, matrix2) => {
  const result = [];
  for (let i = 0; i < matrix1.length; i++) {
    result[i] = 0;
    for (let j = 0; j < matrix2.length; j++) {
      result[i] += matrix1[i][j] * matrix2[j];
    }
  }
  return result;
};

const matrixSubtract = (matrix1, matrix2) => {
  const result = [];
  for (let i = 0; i < matrix1.length; i++) {
    result[i] = matrix1[i] - matrix2[i];
  }
  return result;
};

class RBFNetwork {
  constructor(numInputs, numHidden, numOutputs) {
    this.numInputs = numInputs;
    this.numHidden = numHidden;
    this.numOutputs = numOutputs;

    // Initialize weights and centers randomly
    this.centers = randomMatrix(numHidden, numInputs);
    this.weightsOutput = randomMatrix(numOutputs, numHidden);
  }

  radialBasisFunction(x, c, sigma) {
    const distance = Math.pow(euclideanDistance(x, c), 2);
    return Math.exp(-distance / (2 * sigma * sigma));
  }

  forward(input) {
    const hiddenLayerOutput = [];
    for (let i = 0; i < this.numHidden; i++) {
      const activation = this.radialBasisFunction(input, this.centers[i], 1);
      hiddenLayerOutput[i] = activation;
    }

    const output = matrixMultiply(this.weightsOutput, hiddenLayerOutput);
    return output;
  }

  train(inputs, targets, learningRate, epochs) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i];
        const target = targets[i];

        // Forward pass
        const hiddenLayerOutput = [];
        for (let j = 0; j < this.numHidden; j++) {
          const activation = this.radialBasisFunction(input, this.centers[j], 1);
          hiddenLayerOutput[j] = activation;
        }

        const output = matrixMultiply(this.weightsOutput, hiddenLayerOutput);

        // Backward pass
        const outputError = matrixSubtract(target, output);

        // Update output layer weights
        for (let j = 0; j < this.numOutputs; j++) {
          for (let k = 0; k < this.numHidden; k++) {
            this.weightsOutput[j][k] += learningRate * outputError[j] * hiddenLayerOutput[k];
          }
        }
      }
    }
  }
}

// Example usage to approximate sine function
const network = new RBFNetwork(1, 10, 1);

// Generate training data
const inputs = [];
const targets = [];
for (let i = 0; i < 100; i++) {
  const x = [(Math.PI * 2 * i) / 100]; // Normalize input to [0, 2*pi]
  const y = [Math.sin(x[0])]; // Sinx
  inputs.push(x);
  targets.push(y);
}

// Train the network
network.train(inputs, targets, 0.01, 10000);

// Test the network
const testInput = [(Math.PI * 2 * 0.25) / 100]; // Test input at x = pi/2
const predictedOutput = network.forward(testInput);

console.log(`Target: ${Math.sin(testInput[0])}`);
console.log(`Predicted: ${Math.abs(predictedOutput[0])}`);
