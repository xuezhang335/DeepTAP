from .process import *
from .parse_args import commandLineParser


def deeptap_main():
    args = commandLineParser()
    CurDir = os.path.dirname(os.path.realpath(__file__))
    i = datetime.datetime.now()
    print(f"{i}: Prediction starting ... \n")
    if args.file:
        test_file = pd.read_csv(args.file)
        test_peptide = test_file["peptide"]
        test_data = make_tensordataset(test_peptide)
    else:
        test_peptide = args.peptide
        test_data = make_tensor_single_peptide(test_peptide)
    outDir = args.outputDir if args.outputDir else "."

    predScores = torch.zeros(5, len(test_data))
    for i in range(5):
        model_path = f"{CurDir}/model/{args.taskType}-{i+1}.ckpt"
        checkpoint = torch.load(model_path)
        config = checkpoint["hyper_parameters"]
        model = Model.load_from_checkpoint(model_path, config=config)
        model.eval()
        model.freeze()
        y_hat = model(test_data)
        predScores[i] = torch.squeeze(y_hat)

    pred_score = predScores.mean(dim=0)
    pred_label = [1 if i >= 0.5 else 0 for i in pred_score]

    if args.taskType == 'reg':
        pred_score = score2aff(pred_score)
        pred_label = [1 if i < 10000 else 0 for i in pred_score]

    if args.file:
        file_name = args.file.split('/')[-1].split('.')[0]
        outfile = f'{outDir}/{file_name}_DeepTAP_{args.taskType}_predresult.csv'

        with open(outfile, "w")as f:
            f.write("peptide,pred_score,pred_label\n") if args.taskType == 'cla' else f.write(
                "peptide,pred_affinity,pred_label\n")
            for i, _ in enumerate(test_peptide):
                f.write(
                    f"{test_peptide[i]},{pred_score[i]:.4f},{pred_label[i]}\n")

        outfile_rank = f'{outDir}/{file_name}_DeepTAP_{args.taskType}_predresult_rank.csv'
        predresult = pd.read_csv(outfile)
        predresult_rank = predresult.sort_values(
            by=['pred_score'], ascending=False) if args.taskType == 'cla' else predresult.sort_values(by=['pred_affinity'])
        predresult_rank.to_csv(outfile_rank, index=False)

    else:
        outfile = f"{outDir}/{args.peptide}_DeepTAP_{args.taskType}_predresult.csv"
        with open(outfile, "w")as f:
            f.write("peptide,pred_score,pred_label\n") if args.taskType == 'cla' else f.write(
                "peptide,pred_affinity,pred_label\n")
            f.write(f"{test_peptide},{pred_score[0]:.4f},{pred_label[0]}\n")

    j = datetime.datetime.now()
    print(f"{j}: Prediction end.\n")


if __name__ == "__main__":
    
    deeptap_main()
