import getopt
import sys

import pandas as pd
from keras.models import load_model

from solution.feature_extraction import FeatureExtraction
from solution.prediction import plot_reconstruction_error, predict, prepare_test_df

# Define default threshold
THRESHOLD = 0.04612  # model.ipynb


def print_help():
    msg = """\n Usage: python3 detect.py [OPTIONS]
    -p, --plot: plot recontruction error
    -h: display this help message

    Required options:
    -i, --index: specify Elasticsearch index to search
    -c, --count: specify the number of logs per dataset

    Optional threshold value (default value is set to 0.04612):    
    -t, --threshold: specify the threshold for anomaly detection

    """
    print(msg)


def parse_input_arguments(argv: list) -> dict[str, str | int | float]:
    """
    Parses input arguments and returns their values for detection.
    """
    index_name = ""
    count = 0
    threshold = 0.0
    plot = 0

    opts, _ = getopt.getopt(
        argv, "hi:c:t:p", ["index=", "count=", "threshold=", "plot"]
    )

    if not opts:
        print_help()
        sys.exit()

    opt_names = [opt[0] for opt in opts]

    if (
        "-i" not in opt_names
        and "--index" not in opt_names
        or "-c" not in opt_names
        and "--count" not in opt_names
    ):
        print("\nBoth index name and number of logs to fetch is required.")
        print_help()
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            help()
            sys.exit()
        elif opt in ("-i", "--index"):
            index_name = arg
        elif opt in ("-c", "--count"):
            count = int(arg)
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)
        elif opt in ("-p", "--plot"):
            plot = 1

    return {
        "index_name": index_name,
        "count": count,
        "threshold": threshold,
        "plot": plot,
    }


def detect(argv: list):
    """Detect anomalies in given index.

    Extract all anomalous sample using given threshold value.
    """

    input_args = parse_input_arguments(argv)

    # Assign threshold value or use default one
    threshold = input_args["threshold"] or THRESHOLD
    fe = FeatureExtraction([input_args["index_name"]], input_args["count"])

    # DataFrame used for autoencoder training - used for input data reformatting
    df = pd.read_pickle("resources/data.pkl")
    autoencoder = load_model("model1")

    test_df = fe.parse_logs()

    # Get original dataframe (before preprocessing part)
    fe_val_df = fe.get_validation_dataframe(test_df)

    # Preprocess data
    test_df = fe.preprocess_dataframe(test_df)

    # Reshape test dataframe for autoencoder analysis
    test_df = prepare_test_df(test_df, df)

    # Analyze test data using autoencoder
    reconstruction_error = predict(autoencoder, test_df)

    if input_args["plot"]:
        plot_reconstruction_error(
            reconstruction_error, threshold, (0, max(reconstruction_error) + 0.01)
        )

    print(
        fe_val_df.loc[
            reconstruction_error[reconstruction_error <= threshold].index
        ].to_string()
    )

    print("\nThreshold: ", threshold)
    print(
        "Number of normal samples: ",
        reconstruction_error[reconstruction_error <= threshold].size,
    )
    print(
        "Number of anomalous samples: ",
        reconstruction_error[reconstruction_error > threshold].size,
    )


if __name__ == "__main__":
    detect(sys.argv[1:])
