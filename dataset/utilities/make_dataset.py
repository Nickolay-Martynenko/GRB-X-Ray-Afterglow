import argparse
import os
import SwiftXRTpreprocessing
from SwiftXRTpreprocessing import read_SwiftXRT, make_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_directory", type=str,
					help="the directory to read data from"
)
parser.add_argument("-t", "--target_directory", type=str,
					default=".",
					help="the directory to create dataset in"
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
	if not os.path.isdir(f"{args.target_directory}"):
		os.mkdir(f"{args.target_directory}")
	dataframe.to_csv(f"{args.target_directory}/{dataframe.name}.csv")
print(f"Dataset created in directory '{args.target_directory}/...'")




