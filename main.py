import torch
import taichi as ti
import sys
import argparse


if __name__ == "__main__":
    # 启用这个之后，命令行似乎就不接受没有名字的参数了
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='en')  # ch均可
    parser.add_argument('-p', '--path', default='.')
    parser.add_argument('-o', '--output', default='ocr_result')
    parser.add_argument('-m', '--model', default='gcn')
    args = parser.parse_args()


    if args.model.upper() == 'GCN':
        from models.GCN import main as GCN_main
        GCN_main.main()
    elif args.model.upper() == 'GAT':
        from models.GAT import main as GAT_main
        GAT_main.main()

    elif args.model.upper() == 'GRAPHSAGE':
        from models.GraphSAGE import main as GRAPHSAGE_main
        GRAPHSAGE_main.main()
    elif args.model.upper() == 'GRAND':
        from models.GRAND import main as GRAND_main
        GRAND_main.main()
    elif args.model.upper() == 'FASTGCN':
        from models.FastGCN import main as FASTGCN_main
        FASTGCN_main.main()



