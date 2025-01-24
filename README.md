# ONNX_PARSER
Setup Instructions:
1.Clone the repository:
git clone <https://github.com/LakshmiSaiGeetha/ONNX_PARSER.git>
cd <ONNX_PARSER>

2.Onnx models path:
https://github.com/onnx/models?tab=readme-ov-file

3.Install the required dependencies:

pip install onnx
pip install onnxruntime

4.Update onnx_compiler.py:
4.1. Set the model path, image path, and output directory in the script:
 compiler = onnx_compiler('/path/to/your/onnx/model.onnx')
 compiler.run_inference('/path/to/input/image.jpg')
 compiler.dump('/path/to/output/directory')

5.Compile and run the script:

python3 onnx_compiler.py
