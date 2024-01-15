import * as tf from "@tensorflow/tfjs";

import { TRAINING_DATA } from "./fashion-mnist";

const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS).div(255);
// INPUTS_TENSOR.print();
console.log(INPUTS_TENSOR.dtype)
console.log(INPUTS_TENSOR.shape)


const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// Data is already NORMALIZED and FLATTENED

const model = tf.sequential();
model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

train();

async function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    // validationSplit: 0.15,
    callbacks: { onEpochEnd: logProgress },
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 20,
  });

  function logProgress(epoch, logs) {
    // console.log(logs)
    console.log(`Val Loss for epoch ${epoch}`, Math.sqrt(logs.val_loss));
    console.log(`Accuracy for epoch ${epoch}`, Math.sqrt(logs.val_acc));
  }

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  evaluate();
}
const PREDICTION_ELEMENT = document.getElementById("prediction");


// OF COURSE
// There is big thing to improve:
// Right now we perform a check on a training dataset
// It's obvious it should be performed on a TEST dataset
// or even at least on validation.
// but it's a demo project for simplicity

const labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag','Ankle boot']

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length);
  let answer = tf.tidy(() => {
    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();
    let output = model.predict(newInput);
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = labels[index];
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
  // digit.print();
  console.log(digit)
  digit = tf.tensor(digit, [28, 28], 'float32').div(255);
  tf.browser.toPixels(digit, CANVAS);
  setTimeout(evaluate, 2000);
}
