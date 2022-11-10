#python3 main.py -i data/monvec.csv -o i #Eval monet
#python3 main.py -p data/monet_jpg -s data/chkpoints/exmp1.pt #Train pgan
#python3 main.py -p data/chkpoints/exmp1.pt -o data/tgen/ -m -p#Generate something
#python3 main.py -p data/monet_wds.tar -l data/monet_jpg/ #Create tar ds
#python3 main.py -p 57 -s data/xlt1.pth #Train xla
#python3 main.py -p data/xlt1.pth -o data/ -m x #Generate something
python3 newmain.py train_mul_xla -s data/multr.pth -p 19 #Train muticore
