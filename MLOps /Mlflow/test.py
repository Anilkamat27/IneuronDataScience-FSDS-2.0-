import os


n_estimators =[110,100,150,200,210]
max_depth = [20,25,15,10,5]

for n in n_estimators:
    for m in max_depth:
        os.system(f"python basic_ml_model.py -n {n} -m {m}")




'''import argparse


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--name", "-n",default = "Anil", type = str)
    args.add_argument("--age", "-a",default = 21, type = str)
    parse_args = args.parse_args()
    print(parse_args.name,parse_args.age)'''
