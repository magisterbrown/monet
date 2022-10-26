#python3 main.py -i data/monvec.csv -o i #Eval monet
#python3 main.py -p data/monet_jpg -s data/chkpoints/exmp1.pt #Train pgan
#python3 main.py -p data/chkpoints/exmp1.pt -o data/tgen/ #Generate something
python3 main.py -p data/monet_wds.tar -l data/monet_jpg/ #Create tar ds
