from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2', '-allow-unsupported-compiler']}),
    ],
    cmdclass={'build_ext': BuildExtension})
