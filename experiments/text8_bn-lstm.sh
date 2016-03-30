command="../text8.py --length 180 --initialization orthogonal --optimizer adam --learning-rate 0.01 --batch-size 64 --num-epochs 50"
nh=2000
directory=bn-lstm-nh$nh
mkdir -p $directory
cd $directory
ipython --pdb $command --num-hidden $nh
cd ..
