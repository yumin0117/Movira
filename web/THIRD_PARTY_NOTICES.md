# Movira Web — attribution

Movira application source: AGPL-3.0. Full text: `assets/LICENSE.movira`.

ONNX Runtime Web 1.20.1, Microsoft Corporation: MIT license (`assets/LICENSE.onnxruntime`).
https://github.com/microsoft/onnxruntime/tree/v1.20.1
Distributed browser assets come from the pinned npm `onnxruntime-web@1.20.1` package.

YOLOv8n-pose, Ultralytics; ONNX conversion published by Xenova: AGPL-3.0.
https://huggingface.co/Xenova/yolov8n-pose
Pinned model revision: f0be3feb14d7d9cb44a7ec00ea6cc63899d02158.
SHA-256: 04f6d2416266f2aba6c5ba8b26de33ed9eba3279f972ac02e33f9f1366547586.
https://github.com/ultralytics/ultralytics

DM Sans and Noto Sans KR fonts are requested from Google Fonts under the SIL Open Font License.
Icons, skeleton illustration and interface graphics are drawn in the Movira source.

Foot landmark refinement: OpenCV Zoo MediaPipe Pose, Apache-2.0 (`assets/LICENSE.mediapipe`).
The model predicts 33 body landmarks (plus auxiliary outputs); Movira uses its ankle, heel and foot-index predictions to draw feet.
https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe
File: pose_estimation_mediapipe_2023mar.onnx, distributed locally as `assets/pose-feet.onnx`.
SHA-256: 9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f.
