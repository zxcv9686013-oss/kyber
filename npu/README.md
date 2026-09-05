# KNPU — Kyber NPU Language

KNPU is Kyber's AI accelerator language. It compiles to a DirectML
C++ program that runs neural-network operations on any Windows NPU,
GPU, or CPU via Direct3D 12.

## Block syntax inside a .kyber file

```
#extern "npu"
{
    graph my_inference
        input  float32[1, 3, 224, 224] image
        output float32[1, 1000]        scores

        conv2d     image        -> feat    filters=64  kernel=3  stride=1  padding=1
        relu       feat         -> feat_r
        maxpool    feat_r       -> pooled  kernel=2  stride=2
        flatten    pooled       -> flat
        gemm       flat         -> logits  units=1000
        softmax    logits       -> scores
    end
}
```

## KNPU language reference

### Graph declaration
```
graph <name>
    ...
end
```

### Tensor declarations
```
input  <dtype>[<shape>] <name>
output <dtype>[<shape>] <name>
```
Supported dtypes: float32  float16  int32  uint8

### Operations
| Op | Parameters |
|---|---|
| `conv2d src -> dst` | filters= kernel= stride= padding= |
| `relu src -> dst` | |
| `leaky_relu src -> dst` | alpha= |
| `sigmoid src -> dst` | |
| `tanh src -> dst` | |
| `softmax src -> dst` | axis= |
| `maxpool src -> dst` | kernel= stride= padding= |
| `avgpool src -> dst` | kernel= stride= padding= |
| `gemm src -> dst` | units= bias=true/false |
| `batch_norm src -> dst` | epsilon= momentum= |
| `flatten src -> dst` | |
| `reshape src -> dst` | shape=[N,C,H,W] |
| `add a b -> dst` | |
| `mul a b -> dst` | |
| `concat a b -> dst` | axis= |

### AI hints
```
ai_precision float16    ; run the whole graph in fp16
ai_cache                ; cache compiled graph to disk
ai_target gpu           ; prefer GPU over NPU (default: auto)
ai_target npu
ai_target cpu
```

## Pipeline

```
.npu source  ->  npu_compiler.py  ->  DirectML C++ wrapper
             ->  clang++ / MSVC   ->  .obj
             ->  kyberlink         ->  .exe
```

## Requirements

- Windows 10/11
- DirectX 12 capable GPU or NPU
- DirectML (ships with Windows / WinML)
- clang++ or MSVC cl.exe
- Link with: d3d12.lib dxgi.lib DirectML.lib
