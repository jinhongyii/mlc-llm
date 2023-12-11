import torch
import tvm


@tvm.register_func("torch.topk", override=True)
def torch_topk(x: tvm.nd.NDArray, k: int, out_val: tvm.nd.NDArray, out_indices: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(tvm.nd.array(x, tvm.cpu()))

    out_val_torch_cpu, out_indices_torch_cpu = torch.topk(x_torch.to(torch.float32), k=k)
    out_val_torch_cpu = out_val_torch_cpu.to(x_torch.dtype)
    out_indices_torch_cpu = out_indices_torch_cpu.to(torch.int32)

    out_val.copyfrom(out_val_torch_cpu)
    out_indices.copyfrom(out_indices_torch_cpu)


@tvm.register_func("torch.cumsum", override=True)
def torch_cumsum(x: tvm.nd.NDArray, dim: int, out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(tvm.nd.array(x, tvm.cpu()))
    out_torch_cpu = torch.cumsum(x_torch, dim=dim)
    out.copyfrom(out_torch_cpu)


@tvm.register_func("torch.groupgemm", override=True)
def torch_groupgemm(
    x: tvm.nd.NDArray,
    weight: tvm.nd.NDArray,
    indptr: tvm.nd.NDArray,
    out: tvm.nd.NDArray,
):
    cpu_device = tvm.cpu()
    x_torch = torch.from_dlpack(tvm.nd.array(x, cpu_device)).to("mps")
    weight_torch = torch.from_dlpack(tvm.nd.array(weight, cpu_device)).to("mps")
    indptr_torch = torch.from_dlpack(tvm.nd.array(indptr, cpu_device)).to("mps")

    out_torch = torch.zeros(out.shape, dtype=x_torch.dtype, device="mps")
    assert weight_torch.shape[0] + 1 == indptr_torch.shape[0]
    for i in range(weight_torch.shape[0]):
        out_torch[indptr_torch[i] : indptr_torch[i + 1], :] = torch.matmul(
            x_torch[indptr_torch[i] : indptr_torch[i + 1], :], weight_torch[i].T
        )

    out.copyfrom(out_torch.to("cpu"))
