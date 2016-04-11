command="../text8.py --length 180 --initialization orthogonal --optimizer adam --learning-rate 0.001 --batch-size 128 --num-epochs 50"
nh=2000
directory=text8-bn-lstm-nh$nh
mkdir -p $directory
cd $directory
ipython --pdb $command --num-hidden $nh
cd ..
