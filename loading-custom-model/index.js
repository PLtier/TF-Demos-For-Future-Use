import * as tf from "@tensorflow/tfjs"
const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";

let model = undefined;

async function loadModel(){
    model = await tf.loadLayersModel(MODEL_PATH)
    model.summary()

    //create a batch of 1
    const input = tf.tensor2d([[870]])

    // predict is SYNC!!!
    const inputBatch = tf.tensor2d([[500], [1100], [970]]);
    const result = model.predict(input)
    const resultBatch = model.predict(inputBatch);

    result.print();
    resultBatch.print();

    input.dispose()
    inputBatch.dispose()
    result.dispose()
    resultBatch.dispose();
    model.dispose();
    // MODEL IS A TENSOR AS WELL WHICH NEEDS TO BE DISPOSED!@!
}

loadModel();