# export CUDA_HOME=/cluster/public_datasets/nvidia/cuda-11.3
# export PATH=/cluster/public_datasets/nvidia/cuda-11.3/bin:${PATH}
# export LD_LIBRARY_PATH=/cluster/public_datasets/nvidia/cuda-11.3/lib64:${LD_LIBRARY_PATH}

cd lib/pointnet2
python setup.py install
cd ../../

cd lib/sphericalmap_utils
python setup.py install
cd ../../


pip install gorilla-core==0.2.6.0
pip install opencv-python
pip install gpustat==1.0.0
pip install --upgrade protobuf

