// Define the neural network architecture
const inputLayerSize = 2;
const hiddenLayerSize = 3;
const outputLayerSize = 1;

// Initialize weights and biases with random values
let weightsInputHidden = [...Array(hiddenLayerSize)].map(() =>
  Array(inputLayerSize).fill(Math.random()),
);
let biasesHidden = Array(hiddenLayerSize).fill(0);

let weightsHiddenOutput = [...Array(outputLayerSize)].map(() =>
  Array(hiddenLayerSize).fill(Math.random()),
);
let biasesOutput = Array(outputLayerSize).fill(0);

// Hyperparameters
const learningRate = 0.01;
const epochs = 100000;

// Training data
const trainingData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

// Sigmoid activation function
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Derivative of the sigmoid function
function sigmoidDerivative(x) {
  const sig = sigmoid(x);
  return sig * (1 - sig);
}

// Mean Squared Error loss function
// function calculateError(output, target) {
//   return 0.5 * Math.pow(target - output, 2);
// }

// Backpropagation training
for (let epoch = 0; epoch < epochs; epoch++) {
  for (const data of trainingData) {
    // Forward pass
    const inputLayerOutput = data.input;
    const hiddenLayerInput = weightsInputHidden.map((row, i) =>
      row.reduce((sum, val, j) => sum + val * inputLayerOutput[j], biasesHidden[i]),
    );
    const hiddenLayerOutput = hiddenLayerInput.map(sigmoid);

    const outputLayerInput = weightsHiddenOutput.map((row, i) =>
      row.reduce((sum, val, j) => sum + val * hiddenLayerOutput[j], biasesOutput[i]),
    );
    const networkOutput = outputLayerInput.map(sigmoid)[0];

    // Backward pass (Backpropagation)
    const target = data.output[0];
    // const error = calculateError(networkOutput, target);

    // Compute gradients
    const outputLayerDelta = (networkOutput - target) * sigmoidDerivative(outputLayerInput[0]);
    const hiddenLayerDeltas = hiddenLayerOutput.map(
      (_, i) =>
        outputLayerDelta * weightsHiddenOutput[0][i] * sigmoidDerivative(hiddenLayerInput[i]),
    );

    // Update weights and biases
    for (let i = 0; i < outputLayerSize; i++) {
      biasesOutput[i] -= learningRate * outputLayerDelta;
      for (let j = 0; j < hiddenLayerSize; j++) {
        weightsHiddenOutput[i][j] -= learningRate * outputLayerDelta * hiddenLayerOutput[j];
      }
    }

    for (let i = 0; i < hiddenLayerSize; i++) {
      biasesHidden[i] -= learningRate * hiddenLayerDeltas[i];
      for (let j = 0; j < inputLayerSize; j++) {
        weightsInputHidden[i][j] -= learningRate * hiddenLayerDeltas[i] * inputLayerOutput[j];
      }
    }
  }
}

// Testing the trained network
console.log('Testing the trained network:');
for (const data of trainingData) {
  const inputLayerOutput = data.input;
  const hiddenLayerOutput = weightsInputHidden
    .map((row, i) => row.reduce((sum, val, j) => sum + val * inputLayerOutput[j], biasesHidden[i]))
    .map(sigmoid);
  const networkOutput = weightsHiddenOutput
    .map((row, i) => row.reduce((sum, val, j) => sum + val * hiddenLayerOutput[j], biasesOutput[i]))
    .map(sigmoid)[0];
  console.log(
    `Input: ${data.input}, Target: ${data.output}, Predicted: ${networkOutput.toFixed(4)}`,
  );
}
