command="../penntreebank.py --optimizer adam --initialization orthogonal --num-epochs 50 --length 100 --batch-size 64"
lr=0.002
nh=1000
directory=bn-lstm-nh$nh-lr$lr
mkdir -p $directory
cd $directory
ipython --pdb $command --num-hidden $nh --learning-rate $lr
cd ..
