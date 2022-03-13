from argparse import ArgumentParser


def commandLineParser():
    parser = ArgumentParser()
    print('''
            =====================================================================
            DeepTAP is a deep learning approach used for predicting high-confidence
            
            TAP-binding peptide.
            

            Usage:

            Single peptide:

                deeptap -P [LNIMNKLNI] -O [output directory] 

            List of peptides in a file:

                deeptap -F [file] -O [output directory]  

                (see 1.csv in demo/ for the detailed format of input file)
            =====================================================================
            ''')

    parser.add_argument("-P", "--peptide",
                        help="Single peptide for prediction")
    parser.add_argument("-F", "--file",
                        help="Input file with peptides for prediction: if given, overwrite -P option")
    parser.add_argument("-O", "--outputDir",
                        help="Directory to store file with prediction result: if not given, the current directory will be applied")
    args = parser.parse_args()
    return args
