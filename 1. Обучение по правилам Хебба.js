{
  // AND
  // Define the input patterns for the logical AND operation
  const inputPatterns = [
    [-1, -1], // 0 AND 0
    [-1, 1], // 0 AND 1
    [1, -1], // 1 AND 0
    [1, 1], // 1 AND 1
  ];

  // Define the corresponding target outputs (teacher signals)
  const targetOutputs = [-1, -1, -1, 1];

  // Initialize weights and bias
  let weights = [0, 0];
  let bias = 0;

  // Learning rate
  const learningRate = 1;

  // Number of iterations (epochs)
  const epochs = 10;

  // Train the neural network using Hebb's rule with teacher
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < inputPatterns.length; i++) {
      // Update weights and bias using Hebb's rule with teacher
      weights[0] += learningRate * targetOutputs[i] * inputPatterns[i][0];
      weights[1] += learningRate * targetOutputs[i] * inputPatterns[i][1];
      bias += learningRate * targetOutputs[i];
    }
  }

  for (let i = 0; i < inputPatterns.length; i++) {
    // Weight sum OR NET
    let netInput = inputPatterns[i][0] * weights[0] + inputPatterns[i][1] * weights[1] + bias;
    // Activation function
    let output = netInput >= 0 ? 1 : -1;
    console.log(`Input: ${inputPatterns[i]}, Output: ${output}`);
  }
}

{
  // OR
  // Define the input patterns for the logical AND operation
  const inputPatterns = [
    [-1, -1], // 0 AND 0
    [-1, 1], // 0 AND 1
    [1, -1], // 1 AND 0
    [1, 1], // 1 AND 1
  ];

  // Define the corresponding target outputs (teacher signals)
  const targetOutputs = [-1, 1, 1, 1];

  // Initialize weights and bias
  let weights = [0, 0];
  let bias = 0;

  // Learning rate
  const learningRate = 1;

  // Number of iterations (epochs)
  const epochs = 10;

  // Train the neural network using Hebb's rule with teacher
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < inputPatterns.length; i++) {
      // Update weights and bias using Hebb's rule with teacher
      weights[0] += learningRate * targetOutputs[i] * inputPatterns[i][0];
      weights[1] += learningRate * targetOutputs[i] * inputPatterns[i][1];
      bias += learningRate * targetOutputs[i];
    }
  }

  for (let i = 0; i < inputPatterns.length; i++) {
    let netInput = inputPatterns[i][0] * weights[0] + inputPatterns[i][1] * weights[1] + bias;
    let output = netInput >= 0 ? 1 : -1;
    console.log(`Input: ${inputPatterns[i]}, Output: ${output}`);
  }
}
