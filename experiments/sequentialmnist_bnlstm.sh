command="../../sequential_mnist.py --lstm --num-epoch 200 --learning-rate 0.001  --init id  --batch-size 100 --noise 0.1"
directory=sequentialmnist-bnlstm
mkdir -p $directory
cd $directory
python $command
cd ..
