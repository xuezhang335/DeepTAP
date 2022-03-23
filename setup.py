from setuptools import setup, find_packages

# Adding following code if Requirements install needed.
# import os
# os.system("pip3 install -r requirements.txt")

requires_packages = ["numpy", "datetime", "six", "aiohttp", "certifi", "urllib3", "importlib-metadata", "pyasn1-modules",
                     "pandas", "torch==1.9.1", "pytorch_lightning==1.4.8"]

setup(
    name='DeepTAP',
    version='1.0',
    packages=find_packages(),
    package_data={'deeptap': ['model/*']},
    install_requires=requires_packages,
    url='https://github.com/zhangxue335/DeepTAP',
    author='Zhang Xue',
    author_email='22119130@zju.edu.cn',
    license=open('LICENSE').read(),
    entry_points={'console_scripts': ['deeptap=deeptap.deeptap:main']}
)
