import * as tf from "@tensorflow/tfjs";

const INPUTS = [...Array(20).keys()];

const OUTPUTS = INPUTS.map((key) => key ** 2);

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor1d(INPUTS);

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

INPUTS_TENSOR.print();
OUTPUTS_TENSOR.print();

// Function to take a Tensor and normalize values
// with respect to each column of values contained in that Tensor.

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    // Find the minimum value contained in the Tensor.
    const MIN_VALUES = min || tf.min(tensor, 0);

    // Find the maximum value contained in the Tensor.
    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now subtract the MIN_VALUE from every value in the Tensor
    // And store the results in a new Tensor.
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // Calculate the adjusted values divided by the range size as a new Tensor.
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
}

// Normalize all input feature arrays and then

// dispose of the original non normalized Tensors.

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log("Normalized Values:");
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min Values:");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max Values:");
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
model.summary();

const LEARNING_RATE = 0.01;
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

(async () => {
  await train();
  evaluate();
})();


async function train() {
  model.compile({
    optimizer: OPTIMIZER,
    loss: "meanSquaredError",
  });

  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      // validationSplit: 0.15,
      callbacks: { onEpochEnd: logProgress },
      shuffle: true,
      batchSize: 2,
      epochs: 200,
    }
  );

  function logProgress(epoch, logs) {
    console.log(`Data for epoch ${epoch}`, Math.sqrt(logs.loss));
    console.log(typeof(epoch))
    if(epoch==170){
      OPTIMIZER.setLearningRate(LEARNING_RATE/2)
    }
  }

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log(
    `Final RMSE of training dataset: ${Math.sqrt(
      results.history.loss.slice(-1)
    )}`
  );
}

function evaluate() {
  tf.tidy(() => {
    let newInput = normalize(
      tf.tensor1d([7]),
      // SUPER big surpise
      // APPARENTLY the TFJS doesn't require to specify 
      // batch size if the input shape is [1]
      // it stops working even for [2]
      // so in this case there is no diff. with
      // tf.tensor2d([[7]])!
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );
    // newInput.NORMALIZED_VALUES = newInput.NORMALIZED_VALUES.expandDims()
    console.log(newInput.NORMALIZED_VALUES.shape);
    let output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });

  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();

  model.dispose();
  console.log(tf.memory());
}
// evaluate();
