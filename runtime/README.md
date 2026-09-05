# Kyber hardware emulation runtime

This runtime provides a software execution model for Kyber XASM blocks across
CPU, GPU, NPU, TPU, and LPU targets. It is not a real silicon simulator, but it
is a working software emulation layer for testing code generation, orchestration,
and multi-target execution flow.

## Usage

```bash
python runtime/kyber_runtime.py --target gpu --source sample.xasm
```

or:

```python
from runtime.kyber_runtime import HardwareRuntime

source = '''
target gpu
kernel void add_kernel(float* a, float* b, float* out) {
  out[i] = a[i] + b[i];
}
'''

result = HardwareRuntime('gpu').run(source, inputs={'a': [1, 2, 3], 'b': [4, 5, 6]})
print(result.output)
```

## Supported targets

- cpu: register and memory emulation
- gpu: parallel vector add / thread simulation
- npu: tensor and matmul simulation
- tpu: tiled matrix execution
- lpu: latent pipeline stage emulation
