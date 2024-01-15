import * as tf from "@tensorflow/tfjs";

import { TRAINING_DATA } from "./fashion-mnist";

const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = tf.scalar(min);
    const MAX_VALUES = tf.scalar(max);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return NORMALIZED_VALUES;
  });

  return result;
}
const LOOKUP = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

// Input feature Array is 2 dimensional.

const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// Data is already NORMALIZED and FLATTENED

const model = tf.sequential();
model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 16,
    kernelSize: 3,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

model.add(
  tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);

model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
model.summary();

train();

async function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const RESHAPED_INPUTS=INPUTS_TENSOR.reshape([INPUTS.length, 28,28,1])

  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.15,
    batchSize: 256,
    epochs: 20,
    callbacks: { onEpochEnd: logProgress },
  });

  function logProgress(epoch, logs) {
    // console.log(logs)
    console.log(`Val Loss for epoch ${epoch}`, Math.sqrt(logs.val_loss));
    console.log(`Accuracy for epoch ${epoch}`, Math.sqrt(logs.val_acc));
  }

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  RESHAPED_INPUTS.dispose();
  evaluate();
}
const PREDICTION_ELEMENT = document.getElementById("prediction");

// OF COURSE
// There is big thing to improve:
// Right now we perform a check on a training dataset
// It's obvious it should be performed on a TEST dataset
// or even at least on validation.
// but it's a demo project for simplicity

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length);
  let answer = tf.tidy(() => {
    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0,255);
    let output = model.predict(newInput.reshape([1,28,28,1]));
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = LOOKUP[index];
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

const CANVAS = document.getElementById("canvas");

function drawImage(digit) {
  digit = tf.tensor(digit, [28, 28]).div(255); //it must be normalized!
  tf.browser.toPixels(digit, CANVAS);
  setTimeout(evaluate, 2000);
}
