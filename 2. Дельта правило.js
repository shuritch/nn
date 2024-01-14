/**
 * Вычисление ошибки необходимо для того, чтобы с каждой итерацией обучения уменьшать эту ошибку.
 * В результате весовые коэффициенты находят что-то среднее, так называемый оптимум значений своих весовых коэффициентов,
 * посредством которого, в дальнейшем могут правильно реагировать на похожие входные воздействия.
 */

class DeltaRule {
  constructor(learningRate) {
    this.learningRate = learningRate;
    this.weights = []; // Initialize weights
  }

  train(inputs, target) {
    // 1 служит якорем удерживающим сильный разброс
    const data = [...inputs, 1];

    // Initialize weights if not done already
    if (this.weights.length === 0) {
      this.weights = new Array(data.length).fill(0);
    }

    // Calculate the predicted output
    const predictedOutput = this.predict(data);

    // Calculate the error
    // Difference between expected and actual
    const error = target - predictedOutput;

    // Update weights based on the delta rule
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += this.learningRate * error * data[i];
    }
  }

  // Calculate the predicted output using the current weights
  predict(inputs) {
    let sum = 0;
    for (let i = 0; i < this.weights.length; i++) {
      sum += this.weights[i] * (inputs[i] ?? 1);
    }
    return sum;
  }
}

{
  const deltaRule = new DeltaRule(0.1); // Learning rate = 0.1
  const trainingData = [
    { inputs: [0, 0], target: 0 },
    { inputs: [0, 1], target: 1 },
    { inputs: [1, 0], target: 1 },
    { inputs: [1, 1], target: 1 },
  ];

  // Training the delta rule with the provided data
  for (let i = 0; i < 10000; i++) {
    for (const data of trainingData) {
      deltaRule.train(data.inputs, data.target);
    }
  }
  // Testing the trained model
  console.log('--OR--');
  console.log(deltaRule.predict([0, 0])); // Should be close to: 0
  console.log(deltaRule.predict([0, 1])); // Should be close to: 1
  console.log(deltaRule.predict([1, 0])); // Should be close to: 1
  console.log(deltaRule.predict([1, 1])); // Should be close to: 1
}

{
  const deltaRule = new DeltaRule(0.1); // Learning rate = 0.1
  const trainingData = [
    { inputs: [0, 0], target: 0 },
    { inputs: [0, 1], target: 0 },
    { inputs: [1, 0], target: 0 },
    { inputs: [1, 1], target: 1 },
  ];

  // Training the delta rule with the provided data
  for (let i = 0; i < 10000; i++) {
    for (const data of trainingData) {
      deltaRule.train(data.inputs, data.target);
    }
  }
  // Testing the trained model
  console.log('--AND--');
  console.log(deltaRule.predict([0, 0])); // Should be close to: 0
  console.log(deltaRule.predict([0, 1])); // Should be close to: 0
  console.log(deltaRule.predict([1, 0])); // Should be close to: 0
  console.log(deltaRule.predict([1, 1])); // Should be close to: 1
}

{
  const deltaRule = new DeltaRule(0.1); // Learning rate = 0.1
  const trainingData = [
    { inputs: [0, 0], target: 0 },
    { inputs: [0, 1], target: 1 },
    { inputs: [1, 0], target: 1 },
    { inputs: [1, 1], target: 2 },
    { inputs: [2, 1], target: 3 },
  ];

  // Training the delta rule with the provided data
  for (let i = 0; i < 10000; i++) {
    for (const data of trainingData) {
      deltaRule.train(data.inputs, data.target);
    }
  }
  // Testing the trained model
  console.log('--SUM--');
  console.log(deltaRule.predict([0, 0])); // Should be close to: 0
  console.log(deltaRule.predict([0, 1])); // Should be close to: 1
  console.log(deltaRule.predict([1, 0])); // Should be close to: 1
  console.log(deltaRule.predict([1, 1])); // Should be close to: 2
  console.log(deltaRule.predict([1, 2])); // Should be close to: 3
  console.log(deltaRule.predict([10, 2])); // Should be close to: 12
  console.log(deltaRule.predict([10, -2])); // Should be close to: 8
}
