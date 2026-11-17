param([string]$DataDirectory=(Join-Path ([Environment]::GetFolderPath('Desktop')) 'Movira_test_Video'))
$ErrorActionPreference='Stop';$folder=Join-Path $DataDirectory 'reference-weights';New-Item -ItemType Directory -Path $folder,'web/models' -Force | Out-Null
$detector=Join-Path $folder 'detector.onnx';$pose=Join-Path $folder 'pose.onnx'
if(-not (Test-Path -LiteralPath $detector)){Invoke-WebRequest 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx' -OutFile $detector}
if(-not (Test-Path -LiteralPath $pose)){$zip=Join-Path $folder 'pose-s26.zip';Invoke-WebRequest 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip' -OutFile $zip;Add-Type -AssemblyName System.IO.Compression.FileSystem;$archive=[IO.Compression.ZipFile]::OpenRead($zip);try{$entry=$archive.Entries | Where-Object {$_.FullName.EndsWith('.onnx')} | Select-Object -First 1;[IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$pose,$false)}finally{$archive.Dispose()}}
if((Get-FileHash -LiteralPath $detector -Algorithm SHA256).Hash.ToLower() -ne '427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7'){throw 'Detector weights do not match the pinned source'}
if((Get-FileHash -LiteralPath $pose -Algorithm SHA256).Hash.ToLower() -ne '85c5d669aa0b0c7969799ea9e0620d3b8c1d3c2b6878243e98a1ef8351457986'){throw 'Pose weights do not match the pinned source'}
node scripts/convert-weights.cjs $detector web/models/person
if($LASTEXITCODE -ne 0){throw 'Detector conversion failed'}
node scripts/convert-weights.cjs $pose web/models/pose
if($LASTEXITCODE -ne 0){throw 'Pose conversion failed'}
Invoke-WebRequest 'https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE' -OutFile web/models/LICENSE-YOLOX.txt
Invoke-WebRequest 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/LICENSE' -OutFile web/models/LICENSE-MMPose.txt
