from Code.cae_ocsvm import *
from Code.load_ucsd import *
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--index_flow", type=int, default=15,
                       help="number epoch of flow image ")
    parse.add_argument("--index_model", type=int, default=2,
                       help="1、VAE 2、CAE_FLOW 3、CAE_RGB")
    parse.add_argument("--index_dataset", type=int, default=2,
                       help="0、ped1 1、ped2 2、Avenue")
    parse.add_argument("--epoch", type=int, default=50,
                       help="number of epoch")
    parse.add_argument("--hid_dim", type=int, default=480,
                       help="input latent number of OC-SVM")
    parse.add_argument("--lr", type=float, default=1e-4,
                       help="inital learning rate")
    parse.add_argument("--pattern", type=bool, default=0,
                       help="if pattern=0 cae_ocsvm, else pattern=1 cae_rgb/vae ")
    args = parse.parse_args()

    if args.pattern == 0:
        file_flow = '../checkpoint/CAE/cae_flow_train_' + str(args.index_dataset) + '_' + \
                    str(args.index_model) + '_' + str(args.index_flow) + '.pk'
        if not os.path.exists(file_flow):
            print(file_flow)
            if torch.cuda.is_available():
                    model = Model(args).cuda()
            else:
                model = Model(args)
            print("Flow Train")
            train_data = Flow_Dataset(args.index_dataset,
                                           transform=transforms.ToTensor(),)
            train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=False)
            print("load the flow data done")
            model.cae_flow_train(train_loader)
        else:
            model = Model(args)
            if not os.path.exists('../OC_SVM_File/OneClassSvm_{}_{}.m'.format(args.index_dataset, args.index_model)):
                print("Start train Flow OneClassSvm")
                train_data = Flow_Dataset(args.index_dataset,
                                          transform=transforms.ToTensor())
                train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
                print("load the flow data done")
                model.OneClassSvm(train_loader)
            else:
                print("Start test Flow OneClassSvm")
                if args.index_dataset ==0:
                    labels = load_Ped1_label()
                elif args.index_dataset ==1:
                    labels = load_Ped2_label()
                else:
                    labels = load_Avenue_label()
                model.OneClassSvm_test(labels)
    else:
        file_rgb = '../checkpoint/CAE/cae_rgb_train_' + str(args.index_dataset) + '_' + \
                   str(args.index_model) + '_' + str(args.index_flow) + '.pk'
        if not os.path.exists(file_rgb):
            print(file_rgb)
            if torch.cuda.is_available():
                model = Model(args).cuda()
            else:
                model = Model(args)
            print("RGB Train")
            train_data = RGB_Dataset(args.index_dataset,
                                           transform=transforms.ToTensor(),)
            train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
            print("load the rgb data done")
            # model.vae_train(train_loader)
            model.cae_rgb_train(train_loader)
        else:
            model = Model(args).cuda()
            if not os.path.exists('../OC_SVM_File/OneClassSvm_{}_{}.m'.format(args.index_dataset, args.index_model)):
                print("Start train RGB OneClassSvm")
                train_data = RGB_Dataset(args.index_dataset,
                                          transform=transforms.ToTensor())
                train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=False)
                print("load the rgb data done")
                model.OneClassSvm(train_loader)
            else:
                print("Start test RGB OneClassSvm")
                if args.index_dataset ==0:
                    labels = load_Ped1_label()
                elif args.index_dataset ==1:
                    labels = load_Ped2_label()
                else:
                    labels = load_Avenue_label()
                model.OneClassSvm_test(labels)
