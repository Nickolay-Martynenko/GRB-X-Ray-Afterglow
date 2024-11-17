import argparse
import os
import SwiftXRTpreprocessing
from SwiftXRTpreprocessing import read_SwiftXRT, make_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_directory", type=str,
					help="the directory to read data from"
)
parser.add_argument("-n", "--name", type=str,
					help="the name of output .csv files"
)
parser.add_argument("-p", "--preprocesser", type=str,
					choices=["rebin", "extract_features"],
					default="extract_features",
					help="preprocesser function name",
)
parser.add_argument("-r", "--regime", type=str,
					choices=["padding", "linear_interpolation", "none"],
					default="none",
					help="rebinning regime (ignored unless rebin preprocesser is specified)"
)

args = parser.parse_args()
kwargs = dict()
if args.preprocesser:
	preprocesser = getattr(SwiftXRTpreprocessing, args.preprocesser)
	kwargs.update({"preprocesser": preprocesser})
	if args.preprocesser == "rebin":
		kwargs.update({"preprocesser_kwargs": {"regime": args.regime}})
print("") # print blank line to make output look nice

SwiftXRTdict = read_SwiftXRT(args.source_directory)
print("") # print blank line to make output look nice

train, val, test = make_dataset(SwiftXRTdict, **kwargs)
print("") # print blank line to make output look nice

for dataframe in (train, val, test):
	if not os.path.isdir(f"{dataframe.name}"):
		os.mkdir(f"{dataframe.name}")
	dataframe.to_csv(f"{dataframe.name}/{args.name}.csv")
print(
	f"Dataset created:\n"+
	f".\n"+
	f"└── train\n"+
	f"    └── {args.name}.csv\t({len(train)} entries)\n"+
	f"└── val\n"+
	f"    └── {args.name}.csv\t({len(val)} entries)\n"+
	f"└── test\n"+
	f"    └── {args.name}.csv\t({len(test)} entries)\n"
)




