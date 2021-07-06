from setuptools import setup, find_packages

setup(
    name="onnx-builder",
    version="0.0.2",
    install_requires=["numpy", "onnx"],
    packages=find_packages(),
    url="https://github.com/impactaky/onnx_builder"
)
