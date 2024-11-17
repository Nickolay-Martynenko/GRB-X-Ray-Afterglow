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

SwiftXRTdict = read_SwiftXRT(args.source_directory)
train, val, test = make_dataset(SwiftXRTdict, **kwargs)
for dataframe in (train, val, test):
	if not os.path.isdir(f"{dataframe.name}"):
		os.mkdir(f"{dataframe.name}")
	dataframe.to_csv(f"{dataframe.name}/{args.name}.csv")
print(
	f"Dataset created:\n"+
	f"└── {os.getcwd()}\n"+
	f"    └── train\n"+
	f"        └── {args.name}.csv\n"+
	f"    └── val\n"+
	f"        └── {args.name}.csv\n"+
	f"    └── test\n"+
	f"        └── {args.name}.csv\n"
)




