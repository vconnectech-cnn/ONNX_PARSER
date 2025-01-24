from typing import Any, Sequence,Dict
import numpy as np
import onnx
import onnxruntime
from PIL import Image
from onnx import numpy_helper
from pathlib import Path
import os
import csv


def normalize(img_data) -> np.ndarray:
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def denormalize(img_data) -> np.ndarray:
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    denorm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        denorm_img_data[i,:,:] = (img_data[i,:,:]*stddev_vec[i] + mean_vec[i]) * 255
    return np.rint(denorm_img_data).astype(np.uint8)

def image_to_np(file_name) -> np.ndarray:
    input_image = Image.open(file_name)
    input_image = input_image.resize((224,224))
    img = np.transpose(np.asarray(input_image),axes=(2,1,0))
    img = normalize(img)
    return img

def np_to_image(input_arr) -> Image.Image:
    img = np.transpose(denormalize(input_arr),axes=(2,1,0))
    return Image.fromarray(img)

def heximage_to_np(file_name,shape) -> np.ndarray:
    img = np.empty(shape,dtype=np.uint32)
    img = img.flatten()
    with open('data.hex','r') as f:
        count = 0
        for line in f:
            img[count] = np.uint32(int(line.rstrip(),16))
            count += 1
    return np.reshape(img.view('float32'),shape)

def dump_np_array(path, array):
    Path(os.getcwd() + '/output/' + path).mkdir(parents=True, exist_ok=True)
   
    for key in array:
        arr = array[key]
       
        if np.issubdtype(arr.dtype, np.floating) and arr.dtype == np.float32:
            arr_uint = arr.view(dtype=np.uint32)
        elif np.issubdtype(arr.dtype, np.floating) and arr.dtype == np.float64:
            arr_uint = arr.view(dtype=np.uint64)
        elif np.issubdtype(arr.dtype, np.integer):
            arr_uint = arr
       
        while len(arr_uint.shape) < 4:
            arr_uint = np.expand_dims(arr_uint, axis=0)
       
        num_channels = arr_uint.shape[1] if len(arr_uint.shape) > 1 else 1
        num_kernels = arr_uint.shape[0]
       
        for n in range(num_kernels):
            for c in range(num_channels):
                file_name = f"{os.getcwd()}/output/{path}/{key}_n_{n}_c_{c}.txt"
                np.savetxt(file_name, arr_uint[n, c, :].flat, fmt='%08x', delimiter='\n')
   
    return


def dump_csv(path,nodes,input_data,outputs,initializers):
    Path(os.getcwd()+'/output/'+path).mkdir(parents=True, exist_ok=True)
    with open(os.getcwd()+'/output/'+path+'/params.csv', 'w', newline='') as csvfile:
        header = ['node_name','I-Height','I-Width','Channels','K-Count','K-Height','K-Width','BN_init_size','Input_size','Initalizers_size','Input_address','Kernel_address','Stride','PaddingT','PaddingB','PaddingL','PaddingR','O-Height','O-Width','O-Features']
        writer = csv.DictWriter(csvfile,fieldnames = header)
        writer.writeheader()
        first_node = True
        previous_layer_kernel_offset = 0
        for node in nodes:
            if(nodes[node].op_type == "Conv"):
                input_tensor = input_data if first_node  else outputs[nodes[node].input[0]]
                output_tensor = outputs[nodes[node].output[0]]
                kernel_tensor = initializers[nodes[node].input[1]]
                pads = 0
                strides = 1
                for attribute in nodes[node].attribute:
                    if attribute.name == "pads":
                        pads = attribute.ints[0]
                    if attribute.name == "strides":
                        strides = attribute.ints[0]
                Input_size = (input_tensor.shape[0] if first_node else input_tensor.shape[1]) * (input_tensor.shape[1] if first_node else input_tensor.shape[2]) * (input_tensor.shape[2] if first_node else input_tensor.shape[3])
                Kernel_size = (kernel_tensor.shape[0]) * (kernel_tensor.shape[2]) * kernel_tensor.shape[3]
                Input_address = previous_layer_kernel_offset
                Kernel_address = Input_size + Input_address
                csvdict = {'node_name':node,
                        'Channels':input_tensor.shape[0] if first_node else input_tensor.shape[1],
                        'I-Height':input_tensor.shape[1] if first_node else input_tensor.shape[2],
                        'I-Width':input_tensor.shape[2] if first_node else input_tensor.shape[3],
                        'K-Count':kernel_tensor.shape[0],
                        'K-Height':kernel_tensor.shape[2],
                        'K-Width':kernel_tensor.shape[3],
                        'Input_size' : Input_size,
                        'Initalizers_size' : Kernel_size,
                        'Input_address' : Input_address,
                        'Kernel_address' : Kernel_address,
                        'Stride':strides,
                        'PaddingT':pads,
                        'PaddingB':pads,
                        'PaddingL':pads,
                        'PaddingR':pads,
                        'O-Height':output_tensor.shape[2],
                        'O-Width':output_tensor.shape[3],
                        'O-Features':kernel_tensor.shape[0]
                        }
                writer.writerow(csvdict)
                first_node=False
                previous_layer_kernel_offset = Kernel_address + Kernel_size
            if(nodes[node].op_type == "BatchNormalization"):
                BN_init_size = initializers[nodes[node].input[1]]
                shape_BN_init_size = BN_init_size.shape[0]
                input_tensor = outputs[nodes[node].input[0]]
                output_tensor = outputs[nodes[node].output[0]]
                bn_input_size = input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
                csvdict = {'node_name':node,
                        'Channels':input_tensor.shape[1],
                        'I-Height':input_tensor.shape[2],
                        'I-Width':input_tensor.shape[3],
                        'O-Height':output_tensor.shape[2],
                        'O-Width':output_tensor.shape[3],
                        'O-Features':kernel_tensor.shape[0],
                        'BN_init_size' : shape_BN_init_size,
                        'Initalizers_size' : shape_BN_init_size * 4,
                        'Input_size' : bn_input_size,
                        }
                writer.writerow(csvdict)
            if(nodes[node].op_type == "Relu"):
                input_tensor = outputs[nodes[node].input[0]]
                output_tensor = outputs[nodes[node].output[0]]
                csvdict = {'node_name':node,
                        'Channels':input_tensor.shape[1],
                        'I-Height':input_tensor.shape[2],
                        'I-Width':input_tensor.shape[3],
                        'O-Height':output_tensor.shape[2],
                        'O-Width':output_tensor.shape[3],
                        'O-Features':kernel_tensor.shape[0]
                        }
                writer.writerow(csvdict)
            if(nodes[node].op_type == "MaxPool"):
                input_tensor =  outputs[nodes[node].input[0]]
                output_tensor = outputs[nodes[node].output[0]]
                pads = 0
                strides = 1
                for attribute in nodes[node].attribute:
                    if attribute.name == "pads":
                        pads = attribute.ints[0]
                    if attribute.name == "strides":
                        strides = attribute.ints[0]
                csvdict = {'node_name':node,
                        'Channels':input_tensor.shape[1],
                        'I-Height':input_tensor.shape[2],
                        'I-Width':input_tensor.shape[2],
                        'K-Height':3,
                        'K-Width':3,
                        'Stride':strides,
                        'PaddingT':pads,
                        'PaddingB':pads,
                        'PaddingL':pads,
                        'PaddingR':pads,
                        'O-Height':output_tensor.shape[2],
                        'O-Width':output_tensor.shape[3],
                        'O-Features':kernel_tensor.shape[0]
                        }
                writer.writerow(csvdict)
            if(nodes[node].op_type == "Add"):
                csvdict = {'node_name':node,
                        'Channels':input_tensor.shape[1],
                        'I-Height':input_tensor.shape[2],
                        'I-Width':input_tensor.shape[3],
                        'O-Height':output_tensor.shape[2],
                        'O-Width':output_tensor.shape[3],
                        'O-Features':kernel_tensor.shape[0]
                        }
                writer.writerow(csvdict)
            if(nodes[node].op_type == "GlobalAveragePool"):
                csvdict = {'node_name':node,
                        }
                writer.writerow(csvdict)
            if(nodes[node].op_type == "Flatten"):
                csvdict = {'node_name':node,
                        }
                writer.writerow(csvdict)
            if(nodes[node].op_type == "Gemm"):
                csvdict = {'node_name':node,
                        }
                writer.writerow(csvdict)


    return
    

def infer_node(
    node: onnx.NodeProto,
    input_data : Dict[str,np.ndarray],
    inputs: Dict[str,onnx.ValueInfoProto],
    initializers: Dict[str,onnx.TensorProto],
    opset_ver: int,
    **kwargs: Any,
) -> Sequence[np.ndarray]:
    # Builds the model
    present_outputs = [x for x in node.output if (x != "")]
    

    inputs_vi = [
        inputs[ip] for ip in node.input if ip in inputs
    ]


    for key in input_data:
        if key not in inputs:
            
            inputs_vi.append(onnx.helper.make_tensor_value_info(key,onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_data[key].dtype],input_data[key].shape))
    

    initializer = [
        initializers[init] for init in initializers if init in node.input   
    ]
    outputs_vi = [
            onnx.helper.make_empty_tensor_value_info(output) for output in node.output
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=node.name, inputs=inputs_vi, outputs=outputs_vi, initializer=initializer
    )
    

    schema = onnx.defs.get_schema(node.op_type,opset_ver)
    opset = schema.since_version

    if(node.op_type == "Concat"):
        opset = 11

    kwargs["producer_name"] = "backend-test"
    kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, opset)
    ]
    model = onnx.helper.make_model_gen_version(graph, **kwargs)
    model = onnx.shape_inference.infer_shapes(model)

    options = onnxruntime.SessionOptions()
    #options.enable_profiling=True
    #options.profile_file_prefix = node.name
    options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    providers = [
           ('CUDAExecutionProvider', {
               'device_id': 0,
               'arena_extend_strategy': 'kSameAsRequested',
               'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
               'cudnn_conv_algo_search': 'EXHAUSTIVE',
               'do_copy_in_default_stream': False,
               'cudnn_conv_use_max_workspace': True,
               'cudnn_conv1d_pad_to_nc1d': True
               })
    ] 
    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=providers,
                                        sess_options=options)

    


    feeds = {}
    feeds.update(input_data)
    io_binding = sess.io_binding()
    for ip in feeds:
        io_binding.bind_cpu_input(ip,feeds[ip])
    for op in model.graph.output:
        io_binding.bind_output(name=op.name)
    
    sess.run_with_iobinding(io_binding)
    results = io_binding.copy_outputs_to_cpu()
    #sess.end_profiling()
    return {output.name:results[i] for i,output in enumerate(model.graph.output)}
    



def infer_node_1ch(
    node: onnx.NodeProto,
    input_data : Dict[str,np.ndarray],
    inputs: Dict[str,onnx.ValueInfoProto],
    initializers: Dict[str,onnx.TensorProto],
    opset_ver: int,
    **kwargs: Any,
) -> Sequence[np.ndarray]:
    # Builds the model
    
    if(node.op_type == "Conv"):
        pass
    else:
        results = infer_node(node,input_data,inputs,initializers,opset_ver)
        return results

    inputs_vi = []
    for ip in node.input:
        if ip in inputs:
            shape = (1,1)+(inputs[ip].type.tensor_type.shape.dim[2].dim_value,inputs[ip].type.tensor_type.shape.dim[3].dim_value)
            input_tensor = onnx.helper.make_tensor_value_info(inputs[ip].name,inputs[ip].type.tensor_type.elem_type,shape)
            inputs_vi.append(input_tensor)

    input_1ch = {}
    for key in input_data:
        if(node.op_type == "Conv"):
            input_1ch[key] = np.expand_dims(input_data[key][0][0],axis=(0,1))
            if key not in inputs:
                inputs_vi.append(onnx.helper.make_tensor_value_info(key,onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_1ch[key].dtype],input_1ch[key].shape))


    initializer = [
        initializers[init] for init in initializers if init in node.input
    ]

    initializer_1ch = []

    for init in initializer:
        arr = numpy_helper.to_array(init)
        if(arr.ndim != 1):
            arr = np.expand_dims(arr[0][0],axis=(0,1))
        else:
            arr = np.array(arr[0]).reshape(1)
        initializer_1ch.append(numpy_helper.from_array(arr,init.name))
    

    outputs_vi = [
            onnx.helper.make_empty_tensor_value_info(output) for output in node.output
    ]

    graph = onnx.helper.make_graph(
        nodes=[node], name=node.name, inputs=inputs_vi, outputs=outputs_vi, initializer=initializer_1ch
    )

    schema = onnx.defs.get_schema(node.op_type,opset_ver)
    opset = schema.since_version

    if(node.op_type == "Concat"):
        opset = 11

    kwargs["producer_name"] = "backend-test"
    kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, opset)
    ]
    model = onnx.helper.make_model_gen_version(graph, **kwargs)
    model = onnx.shape_inference.infer_shapes(model)

    options = onnxruntime.SessionOptions()
    #options.enable_profiling=True
    #options.profile_file_prefix = node.name
    options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    providers = [
           ('CUDAExecutionProvider', {
               'device_id': 0,
               'arena_extend_strategy': 'kSameAsRequested',
               'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
               'cudnn_conv_algo_search': 'EXHAUSTIVE',
               'do_copy_in_default_stream': False,
               'cudnn_conv_use_max_workspace': True,
               'cudnn_conv1d_pad_to_nc1d': True
               })
    ] 
    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=providers,
                                        sess_options=options)

    
    feeds = {}
    feeds.update(input_1ch)
    io_binding = sess.io_binding()
    for ip in feeds:
        io_binding.bind_cpu_input(ip,feeds[ip])
    for op in model.graph.output:
        io_binding.bind_output(name=op.name)
    
    sess.run_with_iobinding(io_binding)
    results = io_binding.copy_outputs_to_cpu()
    #sess.end_profiling()
    return {output.name:results[i] for i,output in enumerate(model.graph.output)}
