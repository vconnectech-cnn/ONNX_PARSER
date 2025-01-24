#!/usr/bin/env python
# coding: utf-8

# In[1]:


import onnx
import numpy as np
import onnxruntime
import urllib
import onnx_util


# In[2]:


class onnx_compiler:
    
    def __init__(self,model):
        self.model = onnx.load(model)
        self.input_data = None
        self.graph = self.model.graph
        self.initializers = {initializer.name:initializer for initializer in self.graph.initializer}
        self.inputs = {ip.name:ip for ip in self.graph.input}
        self.outputs = {op.name:op for op in self.graph.output}
        self.nodes = {node.name:node for node in self.graph.node}
        self.intermediate_outputs = {}
        self.single_ch_outputs = {}
        
    def run_inference(self,image):
        print("Onnx verison : ",onnx.__version__)
        self.input_data = onnx_util.image_to_np(image)
        print(hex(self.input_data[2,0,0].view(dtype=np.uint32)))
        for i,node in enumerate(self.graph.node):
            print("Running inference for node %d %s Opset: %d" %(i,node.name,self.model.opset_import[0].version))
            results = {}
            results_single = {}
            if(i == 0):
                results = onnx_util.infer_node(node,
                               {node.input[0]:np.expand_dims(self.input_data,axis=0)},
                               self.inputs,self.initializers,self.model.opset_import[0].version)
                if(node.op_type == "Conv"):
                    results_single = onnx_util.infer_node_1ch(node,
                               {node.input[0]:np.expand_dims(self.input_data,axis=0)},
                               self.inputs,self.initializers,self.model.opset_import[0].version)
                    
            else:
                results = onnx_util.infer_node(node,
                               {name:self.intermediate_outputs[name] for name in node.input if name in self.intermediate_outputs},
                               self.inputs,self.initializers,self.model.opset_import[0].version)
                if(node.op_type == "Conv"):
                    results_single = onnx_util.infer_node_1ch(node,
                               {name:self.intermediate_outputs[name] for name in node.input if name in self.intermediate_outputs},
                               self.inputs,self.initializers,self.model.opset_import[0].version)

            self.intermediate_outputs.update(results)
            self.single_ch_outputs.update(results_single)
            
    def dump(self,model):
        onnx_util.dump_np_array(model+"/input_image",{'data' : self.input_data})
        onnx_util.dump_np_array(model+"/outputs",self.intermediate_outputs)
        onnx_util.dump_np_array(model+"/outputs_1ch",self.single_ch_outputs)
        onnx_util.dump_np_array(model+"/initializers",{name:onnx.numpy_helper.to_array(self.initializers[name]) for name in self.initializers})
        #onnx_util.dump_csv(model + "/csv", self.nodes, self.input_data, self.intermediate_outputs, {name: onnx.numpy_helper.to_array(self.initializers[name]) for name in self.initializers})

        



# In[3]:


compiler = onnx_compiler('/home/admin1/Downloads/vgg16-12.onnx')
compiler.run_inference('/home/admin1/Downloads/dog.jpg')
compiler.dump('VGG-16')

