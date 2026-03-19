"""
TensorRT engine for Metric3D depth estimation.
Loads a serialized .engine file or builds from ONNX, runs inference on GPU.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None
    TENSORRT_AVAILABLE = False


class TensorRTMetric3DEngine:
    """
    TensorRT engine wrapper for Metric3D depth inference.
    Input: tensor (1, 3, H, W) float32, same preprocess as PyTorch Metric3D (resize+pad to 616x1064).
    Output: depth map tensor (1, 1, H, W) or (H, W) float32.
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        device: str = "cuda:0",
        fp16: bool = True,
        workspace_gb: int = 4,
    ):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not installed. Install with: pip install tensorrt"
            )
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.device = device
        self._engine = None
        self._context = None
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._fp16 = fp16
        self._workspace_gb = workspace_gb
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None
        self._stream = None
        self._load_engine()

    def _load_engine(self) -> None:
        if self.engine_path and os.path.isfile(self.engine_path):
            with open(self.engine_path, "rb") as f:
                engine_bytes = f.read()
            runtime = trt.Runtime(self._logger)
            self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        elif self.onnx_path and os.path.isfile(self.onnx_path):
            self._engine = self._build_engine_from_onnx(self.onnx_path)
            if self.engine_path:
                self._save_engine(self._engine, self.engine_path)
        else:
            raise FileNotFoundError(
                f"Neither engine file nor ONNX file found. "
                f"engine_path={self.engine_path}, onnx_path={self.onnx_path}"
            )
        self._context = self._engine.create_execution_context()
        self._infer_binding_info()

    def _build_engine_from_onnx(self, onnx_path: str) -> "trt.ICudaEngine":
        builder = trt.Builder(self._logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self._logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX file")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._workspace_gb << 30)
        if self._fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT engine from ONNX")
        runtime = trt.Runtime(self._logger)
        return runtime.deserialize_cuda_engine(serialized)

    def _save_engine(self, engine: "trt.ICudaEngine", path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(engine.serialize())

    def _infer_binding_info(self) -> None:
        n = self._engine.num_io_tensors
        for i in range(n):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_shape = tuple(shape)
            else:
                self._output_name = name
                self._output_shape = tuple(shape)

    def inference(self, feed_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any, Any]:
        """
        Run inference. Compatible with Metric3D API: model.inference({"input": img_tensor}).
        Returns (pred_depth, None, None) to match model.inference return signature.
        """
        input_tensor = feed_dict["input"]
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_np = input_tensor.contiguous().float().cpu().numpy()
        device_id = torch.device(self.device).index if "cuda" in self.device else 0

        # Set dynamic input shape if needed
        if -1 in (self._input_shape or ()):
            self._context.set_input_shape(self._input_name, input_np.shape)
        else:
            assert input_np.shape == self._input_shape, (
                f"Input shape {input_np.shape} != engine input {self._input_shape}"
            )

        # Allocate output buffer based on context shapes
        out_shape = self._context.get_tensor_shape(self._output_name)
        out_size = int(np.prod(out_shape))
        output_np = np.empty(out_shape, dtype=np.float32)

        # Use torch CUDA tensors for bindings so we can run on correct device
        input_cuda = torch.from_numpy(input_np).to(self.device)
        output_cuda = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        self._context.set_tensor_address(self._input_name, input_cuda.data_ptr())
        self._context.set_tensor_address(self._output_name, output_cuda.data_ptr())
        self._context.execute_async_v3(torch.cuda.current_stream(device_id).cuda_stream)

        torch.cuda.synchronize(device_id)
        pred_depth = output_cuda
        return pred_depth, None, None


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> None:
    """Build and save a TensorRT engine from an ONNX file."""
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed.")
    eng = TensorRTMetric3DEngine(
        engine_path=None,
        onnx_path=onnx_path,
        fp16=fp16,
        workspace_gb=workspace_gb,
    )
    eng._save_engine(eng._engine, engine_path)
    print(f"[OK] Saved TensorRT engine -> {engine_path}")
