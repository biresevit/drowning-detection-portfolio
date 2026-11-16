import argparse
import os
import subprocess
import sys

import tensorrt as trt


def build_engine_fp16_mixed(onnx_path: str, engine_path: str) -> None:
    """Build a TRT engine with FP16, keeping LayerNorm and Softmax in FP32."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    # Allow TRT to fall back to FP32 per layer when we set explicit precision
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Keep precision-sensitive transformer layers in FP32
    fp32_types = {trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX}
    promoted = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in fp32_types:
            layer.precision = trt.DataType.FLOAT
            layer.set_output_type(0, trt.DataType.FLOAT)
            promoted += 1
    print(f"Promoted {promoted} layers to FP32 (LayerNorm + Softmax)")

    # Dynamic batch: min=1, opt=4, max=12; spatial dims fixed at 16x224x224
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        (1,  3, 16, 224, 224),   # min
        (4,  3, 16, 224, 224),   # opt
        (12, 3, 16, 224, 224),   # max
    )
    config.add_optimization_profile(profile)

    print("Building TRT engine (this may take a few minutes)…")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"Engine saved → {engine_path}")


def build_engine_fp32(onnx_path: str, engine_path: str) -> None:
    """Build a standard FP32 TRT engine via trtexec."""
    subprocess.run(
        [
            'trtexec',
            f'--onnx={onnx_path}',
            f'--saveEngine={engine_path}',
            '--minShapes=input:1x3x16x224x224',
            '--optShapes=input:4x3x16x224x224',
            '--maxShapes=input:12x3x16x224x224',
            '--verbose',
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Export VideoSwin to TensorRT engine.')
    parser.add_argument('--model_path', default='/models/drowning_swin_t.pth')
    parser.add_argument('--fp16', action='store_true',
                        help='Build FP16 engine with FP32 LayerNorm/Softmax (mixed precision).')
    args = parser.parse_args()

    model_path = args.model_path
    if os.name == 'nt' and model_path.startswith('/'):
        model_path = model_path[1:]

    onnx_path   = model_path.replace('.pth', '.onnx')
    engine_path = model_path.replace('.pth', '.engine')

    from video_swin import VideoSwin
    model = VideoSwin(
        model_name='swin_t',
        device='cuda',
        classes=['normal', 'drowning'],
        checkpoint_path=model_path,
    )
    model.export_to_onnx(output_path=onnx_path)

    print('Simplifying ONNX model with onnxsim…')
    subprocess.run(
        [sys.executable, '-m', 'onnxsim', onnx_path, onnx_path],
        check=True,
    )

    if args.fp16:
        print('Building mixed-precision FP16 engine (LayerNorm + Softmax in FP32)…')
        build_engine_fp16_mixed(onnx_path, engine_path)
    else:
        print('Building FP32 engine via trtexec…')
        build_engine_fp32(onnx_path, engine_path)


if __name__ == '__main__':
    main()
