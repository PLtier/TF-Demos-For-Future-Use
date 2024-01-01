// import tensorflow
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

console.log(tf.version);

// Add reference to all needed DOM elements
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const enableWebcamButton = document.getElementById("webcamButton");

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!navigator?.mediaDevices?.getUserMedia;
}

if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start classification.
function enableCam(event) {
  // Only continue if the COCO-SSD has finished loading.
  if (!model) {
    return;
  }
  event.target.classList.add("removed");
  const constraints = { video: true };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let model = undefined;

cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  demosSection.classList.remove("invisible");
});

let children = [];

function predictWebcam() {
  model.detect(video).then((predictions) => {
    children.forEach((child) => liveView.removeChild(child));

    children.splice(0);

    predictions.forEach((prediction) => {
      const [x, y, width, height] = prediction.bbox;
      if (prediction.score > 0.66) {
        const p = document.createElement("p");
        p.innerText = `${prediction.class} - with ${Math.round(
          parseFloat(prediction.score) * 100
        )}% confidence`;
        p.style = `margin-left: ${x}px; margin-top: ${y}; width: ${width}px; top: 0; left: 0;`;
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style = `left: ${x}px; top: ${y}px; width: ${width}px; height: ${height}px;`;
        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        children.push(highlighter);
        children.push(p);
      }
      const pre = {
        topLeftCorner: [x, y],
        bottomLeftCorner: [x, y + height],
        topRightCorner: [x + width, y],
        bottomRightCorner4: [x + width, y + height],
        class: prediction.class,
      };
      console.table(pre);
    });
    // window.requestAnimationFrame(predictWebcam);
    setTimeout(predictWebcam, 1000)
  });
}
