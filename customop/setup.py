from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graphop',
    ext_modules=[
        CUDAExtension('graphop', [
            'graphop.cpp',
            'graphop_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
