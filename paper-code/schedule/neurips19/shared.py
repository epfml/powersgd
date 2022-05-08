import re

from jobmonitor.api import upload_code_package


def upload_code():
    code_package, files_uploaded = upload_code_package(
        ".",
        excludes=[
            "core",
            "compression_results",
            "output.tmp",
            ".vscode",
            "node_modules",
            "scripts",
            ".git",
            "bla.tar",
            "*.pyc",
            "._*",
            "__pycache__",
            "*.pdf",
            "*.js",
            "*.yaml",
            ".pylintrc",
            ".gitignore",
            ".AppleDouble",
            ".jobignore",
        ],
    )
    print("Uploaded {} files.".format(len(files_uploaded)))

    return code_package


def sgd_config(lr, momentum, weight_decay):
    return {
        "optimizer_learning_rate": lr,
        "optimizer_conv_learning_rate": lr,
        "optimizer_weight_decay_other": weight_decay,
        "optimizer_weight_decay_conv": weight_decay,
        "optimizer_momentum": momentum,
    }


def language_modeling_base():
    return {
        "task": "LanguageModeling",
        "optimizer_weight_decay_other": 0,
        "optimizer_weight_decay_conv": 0,
        "optimizer_batch_size": 64,
        "optimizer_learning_rate": 20,
        "optimizer_decay_at_epochs": [60, 80],
        "optimizer_momentum": 0,
        "num_epochs": 90,
    }


def superresolution_base():
    return {
        "task": "SuperResolution",
        "optimizer_weight_decay_other": 0,
        "optimizer_weight_decay_conv": 0,
        "optimizer_batch_size": 25,
        "optimizer_learning_rate": 0.1,
        "optimizer_decay_at_epochs": [60, 80],
        "optimizer_momentum": 0.9,
        "num_epochs": 90,
    }


def optimizer_config(reducer):
    match = re.match("Rank (\d+) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_reducer_n_power_iterations": 0,
            "optimizer_reducer_reuse_query": True,
            "optimizer_reducer_rank": rank,
            "optimizer_memory": True,
            "optimizer_reducer": "RankKReducer",
            "reducer_name": reducer,
        }

    match = re.match("Shuffled rank (\d+) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_reducer_n_power_iterations": 0,
            "optimizer_reducer_reuse_query": True,
            "optimizer_reducer_rank": rank,
            "optimizer_memory": True,
            "optimizer_reducer": "ShuffledRankKReducer",
            "reducer_name": reducer,
        }

    match = re.match("HQ Rank (\d+) w/o reuse \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        if rank == 1:
            return {
                "optimizer_reducer_n_power_iterations": 2,
                "optimizer_reducer_reuse_query": False,
                "optimizer_memory": True,
                "optimizer_reducer": "FasterRank1Reducer",
                "reducer_name": reducer,
            }
        else:
            return {
                "optimizer_reducer_n_power_iterations": 2,
                "optimizer_reducer_reuse_query": False,
                "optimizer_reducer_rank": rank,
                "optimizer_memory": True,
                "optimizer_reducer": "RankKReducer",
                "reducer_name": reducer,
            }

    match = re.match("Half Rank (\d+) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_reducer_rank": rank,
            "optimizer_memory": True,
            "optimizer_reducer": "HalfRankKReducer",
            "reducer_name": reducer,
        }

    match = re.match("Unbiased Rank (\d+)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_reducer_rank": rank,
            "optimizer_memory": False,
            "optimizer_reducer_n_power_iterations": 0,
            "optimizer_reducer": "HalfRankKReducer",
            "reducer_name": reducer,
        }

    match = re.match("SVD Rank (\d+) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_reducer_n_power_iterations": 0,
            "optimizer_reducer_reuse_query": True,
            "optimizer_reducer_rank": rank,
            "optimizer_reducer": "SVDReducer",
            "reducer_name": reducer,
        }

    if reducer == "Top K (EF)":
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "TopKReducer",
            "optimizer_reducer_compression": 1 / 244,
            "reducer_name": reducer,
        }

    if reducer == "SGD":
        return {
            "optimizer_memory": False,
            "optimizer_reducer": "ExactReducer",
            "reducer_name": reducer,
        }

    if reducer == "Signum":
        return {
            "optimizer_memory": False,
            "optimizer_reducer": "SignSGDwithMajorityVoteReducer",
            "optimizer_mom_before_reduce": True,
            "optimizer_wd_before_reduce": False,
            "optimizer_momentum_type": "exponential_moving_average",
            "reducer_name": reducer,
        }

    match = re.match("Atomo \(rank (\d+)\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_memory": False,
            "optimizer_reducer": "AtomoReducer",
            "optimizer_reducer_rank": rank,
            "reducer_name": reducer,
        }

    if reducer == "Sign+Norm (EF)":
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "SignAndNormReducer",
            "reducer_name": reducer,
        }

    if reducer == "Sign (EF)":
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "SignReducer",
            "reducer_name": reducer,
        }

    match = re.match("Random K \((\d+)x\) \(EF\)", reducer)
    if match:
        compression_factor = int(match.group(1))
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "UniformRandomSparseReducer",
            "optimizer_reducer_compression": 1 / compression_factor,
            "reducer_name": reducer,
        }

    match = re.match("Random Block \((\d+)x\) \(EF\)", reducer)
    if match:
        compression_factor = int(match.group(1))
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "UniformRandomSparseBlockReducer",
            "optimizer_reducer_compression": 1 / compression_factor,
            "reducer_name": reducer,
        }

    match = re.match("Random K \(rank (\d+)\) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "RandomSparseReducer",
            "optimizer_reducer_rank": rank,
            "reducer_name": reducer,
        }

    match = re.match("Random Block \(rank (\d+)\) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "RandomSparseBlockReducer",
            "optimizer_reducer_rank": rank,
            "reducer_name": reducer,
        }

    match = re.match("Top K \((\d+)x\) \(EF\)", reducer)
    if match:
        compression_factor = int(match.group(1))
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "TopKReducer",
            "optimizer_reducer_compression": 1 / compression_factor,
            "reducer_name": reducer,
        }

    match = re.match("Top K \(rank (\d+)\) \(EF\)", reducer)
    if match:
        rank = int(match.group(1))
        return {
            "optimizer_memory": True,
            "optimizer_reducer": "StratifiedTopKReducer",
            "optimizer_reducer_rank": rank,
            "reducer_name": reducer,
        }
