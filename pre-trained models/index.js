import * as tf from "@tensorflow/tfjs";

const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
const EXAMPLE_IMG = document.getElementById('exampleImg');


let movenet = undefined;

async function loadAndRunModel(){
    movenet = await tf.loadGraphModel(MODEL_PATH, {fromTFHub: true})
    // let exampleInputTensor = tf.zeros([1,192,192,3], 'int32')
    // let tensorOutput = movenet.predict(exampleInputTensor)
    // let arrayOutput = await tensorOutput.array();
    // console.log(arrayOutput);

    let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);

    let cropStartingPoint = [15,170,0]
    let cropSize = [345,345,3]
    let croppedTensor = tf.slice(imageTensor,cropStartingPoint, cropSize)

    let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192,192], true).toInt();
    console.log(resizedTensor.shape)

    let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
    let arrayOutput = await tensorOutput.array();
    console.log(arrayOutput);
}

loadAndRunModel();
