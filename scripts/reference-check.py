"""Development-only numerical oracle. Not part of the browser application."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
root=Path(__file__).resolve().parents[1]
weights=Path.home()/'Desktop'/'Movira_test_Video'/'reference-weights'
records=[]
for name,source,shape in [('person','detector.onnx',(1,3,416,416)),('pose','pose.onnx',(1,3,256,192))]:
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(weights/source),options,providers=['CPUExecutionProvider'])
    data=np.fromfile(root/'web'/'.dev'/f'{name}.input.bin',dtype=np.float32).reshape(shape)
    outputs=session.run(None,{session.get_inputs()[0].name:data})
    record={'name':name,'outputs':[]}
    for info,value in zip(session.get_outputs(),outputs):
        file=f'{name}.{info.name}.expected.bin';value.astype(np.float32).tofile(root/'web'/'.dev'/file)
        record['outputs'].append({'name':info.name,'shape':list(value.shape),'file':file})
    records.append(record)
(root/'web'/'.dev'/'cases.json').write_text(json.dumps(records))
print(json.dumps(records))
