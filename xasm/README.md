# XASM — Multi-target accelerated assembly

XASM is Kyber's AI-augmented assembly language and now supports multiple
backend families in one source file. A single XASM block can target CPU, GPU,
NPU, TPU, or LPU by using `target <backend>` sections, and the compiler will
lower the matching section to the right output format.

## Target sections

```xasm
target cpu
proc add_vectors(float32* a, float32* b, float32* out, int32 n)
    ai_vectorize
    loop i in 0..64
        load rax, float32* rcx[rax]
        load rbx, float32* rdx[rax]
        add  rax, rbx
        store float32* r8[rax], rax
    end
end


target gpu
kernel void add_kernel(float* a, float* b, float* out) {
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}


target npu
ai_target npu
matmul fp16 x y -> z
```

## Supported targets

- cpu -> NASM x86-64 object assembly
- gpu -> PTX-style GPU code / NGPU-style kernel syntax
- npu -> DirectML-oriented backend stub
- tpu -> TPU backend lowering
- lpu -> LPU backend lowering
- cuda -> alias for gpu

## C/C++ interop

XASM can be mixed with C/C++ by placing the XASM block in a source file or
linking generated object files alongside C/C++ objects. The compiler emits
backend objects that can be linked with standard toolchains.

## AI directives

| Directive | Effect |
|---|---|
| `ai_vectorize` | Auto-apply vectorization hints to the next loop |
| `ai_optimize speed` | Prefer fast instruction sequences |
| `ai_optimize size` | Prefer compact instruction sequences |
| `ai_unroll N` | Unroll the next loop N times |
| `ai_inline procname` | Mark a proc for inlining |

## Pipeline

```
.xasm -> xasm_compiler.py -> CPU: NASM -> .obj
                       -> GPU: PTX/NGPU backend -> .obj
                       -> NPU: DirectML backend -> .obj
                       -> TPU/LPU: accelerator backend -> .obj
```

## Installation notes

- NASM is required for CPU assembly: https://nasm.us
- Set `KYBER_NASM` to override the NASM path
- CUDA/DirectML toolchain hints are auto-detected when present
