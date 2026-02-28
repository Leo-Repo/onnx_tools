from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnx_tool
from onnx_tool.fusion import FusionPattern, createSerialPattern
from onnx_tool.node import ConvNode, MaxPoolNode, NODE_REGISTRY, ResizeNode as BaseResizeNode, TmpNodeProto, create_node
from onnx_tool.tensor import Tensor, volume


INPUT_MODEL = Path(r"D:\projects\AIInfra\onnx-tool\data\public\yolov5\yolov5s_noPostprocess.onnx")
OUTPUT_MODEL = Path(r"D:\projects\AIInfra\onnx-tool\data\public\yolov5\yolov5s_noPostprocess_optimized.onnx")
OUTPUT_MODEL_DEDUP = Path(r"D:\projects\AIInfra\onnx-tool\data\public\yolov5\yolov5s_noPostprocess_optimized_dedup.onnx")
MODEL_INPUT_NAME = "images"
VERIFY_INPUT_SHAPE = (1, 3, 640, 640)


ConvSiluPattern = [
    {
        "name": "conv_0",
        "op": "Conv",
        "attrs": [],
        "inport": [],
        "outport": [[0, "sigmoid_0", 0], [0, "mul_0", 0]],
    },
    {
        "name": "sigmoid_0",
        "op": "Sigmoid",
        "attrs": [],
        "inport": [[0, "conv_0", 0]],
        "outport": [[0, "mul_0", 1]],
    },
    {
        "name": "mul_0",
        "op": "Mul",
        "attrs": [],
        "inport": [[0, "conv_0", 0], [1, "sigmoid_0", 0]],
        "outport": [],
    },
]


def register_fused_ops() -> None:
    @NODE_REGISTRY.register()
    class ResizeNode(BaseResizeNode):
        def value_infer(self, intensors: list[Tensor], outtensors: list[Tensor]):
            self.shape_infer(intensors, outtensors)
            x = intensors[0].get_numpy()
            outshape = tuple(int(v) for v in outtensors[0].get_shape())
            if len(outshape) == 0:
                outtensors[0].update_tensor(x)
                return
            indexers = []
            for in_dim, out_dim in zip(x.shape, outshape):
                if out_dim <= 0:
                    indexers.append(np.array([0], dtype=np.int64))
                elif in_dim == out_dim:
                    indexers.append(np.arange(out_dim, dtype=np.int64))
                else:
                    scale = float(in_dim) / float(out_dim)
                    idx = np.floor(np.arange(out_dim, dtype=np.float64) * scale).astype(np.int64)
                    idx = np.clip(idx, 0, in_dim - 1)
                    indexers.append(idx)
            y = x[np.ix_(*indexers)]
            outtensors[0].update_tensor(y.astype(x.dtype, copy=False))

    @NODE_REGISTRY.register()
    class Conv_MaxPoolNode(ConvNode):
        def __init__(self, n):
            super().__init__(n)
            for key in ("auto_pad", "pads", "strides", "dilations", "group"):
                prefixed = f"Conv0_{key}"
                if hasattr(self, prefixed):
                    self.set_attr(key, getattr(self, prefixed))
                elif hasattr(self, key):
                    self.set_attr(key, getattr(self, key))

            self.pool_kernel_shape = tuple(getattr(self, "MaxPool1_kernel_shape", (3, 3)))
            self.pool_ceil_mode = int(getattr(self, "MaxPool1_ceil_mode", 0))
            self.pool_pads = tuple(getattr(self, "MaxPool1_pads", (0, 0, 0, 0)))
            self.pool_strides = tuple(getattr(self, "MaxPool1_strides", (1, 1)))
            self.pool_dilations = tuple(getattr(self, "MaxPool1_dilations", (1, 1)))
            self.pool_auto_pad = getattr(self, "MaxPool1_auto_pad", None)

        def _pool_node(self) -> MaxPoolNode:
            attrs = {
                "kernel_shape": self.pool_kernel_shape,
                "ceil_mode": self.pool_ceil_mode,
                "pads": self.pool_pads,
                "strides": self.pool_strides,
                "dilations": self.pool_dilations,
                "auto_pad": self.pool_auto_pad,
            }
            proto = TmpNodeProto(self.name + "_pool", "MaxPool", attrs, domain=self.domain)
            return MaxPoolNode(proto)

        def shape_infer(self, intensors: list[Tensor], outtensors: list[Tensor]):
            conv_out = Tensor(self.name + "_conv_tmp")
            super().shape_infer(intensors, [conv_out])
            self._pool_node().shape_infer([conv_out], outtensors)

        def value_infer(self, intensors: list[Tensor], outtensors: list[Tensor]):
            conv_out = Tensor(self.name + "_conv_tmp")
            super().value_infer(intensors, [conv_out])
            self._pool_node().value_infer([conv_out], outtensors)

        def profile(self, intensors: list[Tensor], outtensors: list[Tensor]):
            conv_out = Tensor(self.name + "_conv_tmp")
            super().shape_infer(intensors, [conv_out])
            pool_out = Tensor(self.name + "_pool_tmp")
            self._pool_node().shape_infer([conv_out], [pool_out])
            conv_macs, params = super().profile(intensors, [conv_out])
            kernel_vol = volume(list(self.pool_kernel_shape))
            pool_macs = volume(pool_out.get_shape()) * kernel_vol
            return [conv_macs + pool_macs, params]


def _replace_consumer(graph, tensor_name: str, old_consumer: str, new_consumer: str) -> None:
    consumers = graph.consumedby.get(tensor_name, [])
    if old_consumer in consumers:
        consumers.remove(old_consumer)
    if new_consumer not in consumers:
        consumers.append(new_consumer)
    graph.consumedby[tensor_name] = consumers


def _remove_consumer(graph, tensor_name: str, consumer_name: str) -> None:
    if tensor_name not in graph.consumedby:
        return
    consumers = graph.consumedby[tensor_name]
    if consumer_name in consumers:
        consumers.remove(consumer_name)
    if len(consumers) == 0:
        graph.consumedby.pop(tensor_name, None)


def _identity_conv_weights(channels: int) -> np.ndarray:
    w = np.zeros((channels, 1, 1, 1), dtype=np.float32)
    w[:, 0, 0, 0] = 1.0
    return w


def insert_identity_conv_before_maxpool(graph) -> list[str]:
    inserted = []
    maxpool_names = [n for n in list(graph.nodemap.keys()) if graph.nodemap[n].op_type == "MaxPool"]

    for pool_name in maxpool_names:
        pool_node = graph.nodemap[pool_name]
        if len(pool_node.input) == 0:
            continue
        src_tensor = pool_node.input[0]
        if src_tensor not in graph.tensormap:
            continue
        src_shape = graph.tensormap[src_tensor].get_shape()
        if len(src_shape) < 2:
            continue
        channels = int(src_shape[1])

        conv_name = f"{pool_name}_IdentityConv"
        conv_w_name = f"{conv_name}_W"
        conv_out_name = f"{conv_name}_out"

        graph.add_initial(conv_w_name, _identity_conv_weights(channels))
        conv_proto = onnx.helper.make_node(
            "Conv",
            [src_tensor, conv_w_name],
            [conv_out_name],
            name=conv_name,
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
            group=channels,
        )
        conv_node = create_node(conv_proto)
        conv_node.input = [src_tensor, conv_w_name]
        conv_node.output = [conv_out_name]

        graph.nodemap[conv_name] = conv_node
        if conv_out_name not in graph.tensormap:
            graph.tensormap[conv_out_name] = Tensor(conv_out_name)
        if conv_out_name not in graph.dynamics:
            graph.dynamics.append(conv_out_name)

        _replace_consumer(graph, src_tensor, pool_name, conv_name)
        graph.producedby[conv_out_name] = [conv_name]
        graph.consumedby[conv_out_name] = [pool_name]
        pool_node.input[0] = conv_out_name

        inserted.append(conv_name)

    graph.graph_reorder_nodes()
    return inserted


def fuse_conv_silu(graph) -> int:
    pattern = FusionPattern(ConvSiluPattern)
    node_sets = pattern.search_pattern(graph)
    for nodes in node_sets:
        graph.fuse_subgraph_node_names(
            nodes,
            nodeop="Conv_Silu",
            nodename=nodes[0],
            keep_attr=True,
            nodedomain="ai.custom",
        )
    graph.graph_reorder_nodes()
    return len(node_sets)


def fuse_inserted_conv_maxpool(graph, inserted_conv_names: Iterable[str]) -> int:
    inserted = set(inserted_conv_names)
    pattern = createSerialPattern(["Conv", "MaxPool"])
    node_sets = pattern.search_pattern(graph)

    fused = 0
    for nodes in node_sets:
        conv_name = nodes[0]
        if conv_name not in inserted:
            continue
        fused_name = f"{nodes[1]}_ConvMaxPool"
        graph.fuse_subgraph_node_names(
            nodes,
            nodeop="Conv_MaxPool",
            nodename=fused_name,
            keep_attr=True,
            nodedomain="ai.custom",
        )
        fused += 1
    graph.graph_reorder_nodes()
    return fused


def remove_resize_sizes_branches(graph) -> int:
    removed_branches = 0
    resize_names = [n for n in list(graph.nodemap.keys()) if graph.nodemap[n].op_type == "Resize"]

    for resize_name in resize_names:
        resize_node = graph.nodemap[resize_name]
        if len(resize_node.input) < 4:
            continue

        x_tensor_name = resize_node.input[0]
        scales_tensor_name = resize_node.input[2]
        sizes_tensor = resize_node.input[3]
        if x_tensor_name not in graph.tensormap or sizes_tensor not in graph.tensormap:
            continue

        in_shape = np.asarray(graph.tensormap[x_tensor_name].get_shape(), dtype=np.float32)
        sizes_np = np.asarray(graph.tensormap[sizes_tensor].get_numpy(), dtype=np.float32).reshape(-1)
        if in_shape.size != sizes_np.size or np.any(in_shape <= 0):
            continue
        scales_np = (sizes_np / in_shape).astype(np.float32)
        _set_constant_tensor_value(graph, scales_tensor_name, scales_np)

        resize_node.input = resize_node.input[:3]
        _remove_consumer(graph, sizes_tensor, resize_name)

        if sizes_tensor in graph.producedby:
            producer_names = list(graph.producedby[sizes_tensor])
            for pname in producer_names:
                if pname in graph.nodemap and graph.nodemap[pname].shape_calc:
                    graph.remove_subtree(pname)
                    removed_branches += 1

    prune_nodes_not_contributing_to_outputs(graph)
    graph.update_tensor_relations()
    graph.update_graph()
    graph.graph_reorder_nodes()
    return removed_branches


def _set_constant_tensor_value(graph, tensor_name: str, value: np.ndarray) -> None:
    graph.tensormap[tensor_name].update_tensor(value)
    for pname in graph.producedby.get(tensor_name, []):
        pnode = graph.nodemap.get(pname)
        if pnode is None or pnode.op_type != "Constant":
            continue
        pnode.set_attr("value", value)


def prune_nodes_not_contributing_to_outputs(graph) -> int:
    required_tensors = set(graph.output)
    required_nodes = set()
    queue = list(required_tensors)

    while queue:
        tname = queue.pop()
        for producer in graph.producedby.get(tname, []):
            if producer in required_nodes:
                continue
            required_nodes.add(producer)
            pnode = graph.nodemap.get(producer)
            if pnode is None:
                continue
            for inp in pnode.input:
                if inp not in required_tensors:
                    required_tensors.add(inp)
                    queue.append(inp)

    remove_names = [n for n in list(graph.nodemap.keys()) if n not in required_nodes]
    for n in remove_names:
        graph.remove_node(n)
    return len(remove_names)


@dataclass
class VerifyResult:
    allclose: bool
    per_output_max_abs_diff: list[float]
    method: str


def verify_outputs_equal_ort(original_model_path: Path, optimized_model_path: Path, seed: int = 0) -> VerifyResult:
    import onnxruntime as ort

    sess0 = ort.InferenceSession(str(original_model_path), providers=["CPUExecutionProvider"])
    sess1 = ort.InferenceSession(str(optimized_model_path), providers=["CPUExecutionProvider"])

    in0 = sess0.get_inputs()[0]
    ishape = [d if isinstance(d, int) and d > 0 else VERIFY_INPUT_SHAPE[i] for i, d in enumerate(in0.shape)]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(tuple(ishape), dtype=np.float32)

    y0 = sess0.run(None, {in0.name: x})
    y1 = sess1.run(None, {sess1.get_inputs()[0].name: x})

    assert len(y0) == len(y1), "Output count mismatch between original and optimized models."
    max_abs = []
    ok = True
    for a, b in zip(y0, y1):
        diff = float(np.max(np.abs(a - b)))
        max_abs.append(diff)
        if not np.allclose(a, b, rtol=1e-4, atol=1e-4):
            ok = False
    return VerifyResult(ok, max_abs, "onnxruntime")


def ensure_custom_domain_opset(model_path: Path, domain: str = "ai.custom", version: int = 1) -> None:
    model = onnx.load(str(model_path))
    has_domain = any(op.domain == domain for op in model.opset_import)
    if not has_domain:
        model.opset_import.append(onnx.helper.make_opsetid(domain, version))
        onnx.save(model, str(model_path))


def _attr_to_pyval(attr):
    val = onnx.helper.get_attribute_value(attr)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, tuple):
        return list(val)
    return val


def deduplicate_fused_attributes(input_model_path: Path, output_model_path: Path) -> int:
    model = onnx.load(str(input_model_path))
    removed = 0
    conv_keys = {"dilations", "group", "kernel_shape", "pads", "strides"}

    for node in model.graph.node:
        if node.op_type not in ("Conv_Silu", "Conv_MaxPool"):
            continue
        attrs = list(node.attribute)
        has_prefixed = {a.name[len("Conv0_") :] for a in attrs if a.name.startswith("Conv0_")}
        keep = []
        for a in attrs:
            # Keep source-disambiguated names and remove ambiguous duplicates.
            if a.name in conv_keys and a.name in has_prefixed:
                removed += 1
                continue
            keep.append(a)
        del node.attribute[:]
        node.attribute.extend(keep)

    onnx.save(model, str(output_model_path))
    return removed


def optimize_yolov5(
    input_model: Path = INPUT_MODEL,
    output_model: Path = OUTPUT_MODEL,
    output_model_dedup: Path = OUTPUT_MODEL_DEDUP,
) -> None:
    register_fused_ops()

    optimized_model = onnx_tool.Model(str(input_model))

    optimized_graph = optimized_model.graph
    dummy = np.zeros(VERIFY_INPUT_SHAPE, dtype=np.float32)
    optimized_graph.shape_infer({MODEL_INPUT_NAME: dummy})

    inserted_convs = insert_identity_conv_before_maxpool(optimized_graph)
    optimized_graph.shape_infer({MODEL_INPUT_NAME: dummy})
    optimized_graph = optimized_graph.get_compute_graph()
    removed_shape_branches = remove_resize_sizes_branches(optimized_graph)

    std_model_path = output_model.with_name(output_model.stem + "_std.onnx")
    optimized_model.graph = optimized_graph
    optimized_model.save_model(str(std_model_path), no_shape=True)
    verify = verify_outputs_equal_ort(input_model, std_model_path)

    fused_conv_silu = fuse_conv_silu(optimized_graph)
    fused_conv_maxpool = fuse_inserted_conv_maxpool(optimized_graph, inserted_convs)

    optimized_model.graph = optimized_graph
    output_model.parent.mkdir(parents=True, exist_ok=True)
    optimized_model.save_model(str(output_model), no_shape=True)
    ensure_custom_domain_opset(output_model)
    removed_dup_attrs = deduplicate_fused_attributes(output_model, output_model_dedup)
    ensure_custom_domain_opset(output_model_dedup)

    print(f"Input model:  {input_model}")
    print(f"Output model: {output_model}")
    print(f"Output model (dedup attrs): {output_model_dedup}")
    print(f"Fused Conv+Silu: {fused_conv_silu}")
    print(f"Inserted 1x1 identity Conv before MaxPool: {len(inserted_convs)}")
    print(f"Fused Conv+MaxPool: {fused_conv_maxpool}")
    print(f"Removed Resize sizes branches: {removed_shape_branches}")
    print(f"Output match ({verify.method}): {verify.allclose}")
    print(f"Per-output max abs diff: {verify.per_output_max_abs_diff}")
    print(f"Removed duplicated fused attributes: {removed_dup_attrs}")


if __name__ == "__main__":
    optimize_yolov5()
