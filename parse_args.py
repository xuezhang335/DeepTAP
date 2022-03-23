from argparse import ArgumentParser


def commandLineParser():
    parser = ArgumentParser()
    print('''
            =====================================================================
            DeepTAP is a deep learning approach used for predicting high-confidence
            
            TAP-binding peptide.
            

            Usage:

            Single peptide:

                python deeptap.py -t cla -p [LNIMNKLNI] -o [output directory] 

            List of peptides in a file:

                python deeptap.py -t cla -f [file] -o [output directory]  

                (see 1.csv in demo/ for the detailed format of input file)
            =====================================================================
            ''')
    parser.add_argument("-t", "--taskType", default='cla',
                        choices=['cla', 'reg'], help="Select task type: classification, regression")
    parser.add_argument("-p", "--peptide",
                        help="Single peptide for prediction")
    parser.add_argument("-f", "--file",
                        help="Input file with peptides for prediction: if given, overwrite -p option")
    parser.add_argument("-o", "--outputDir",
                        help="Directory to store file with prediction result: if not given, the current directory will be applied")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    commandLineParser()
