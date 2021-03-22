import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="ALD")

    #General
    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'ALD'],
                        help="Experiment Baseline")
    parser.add_argument("--Baseline_num",
                        type=int,
                        default=0,
                        help="Number of baselines")
    parser.add_argument("--CPU_weight",
                        type=float,
                        default=1.,
                        help="The weight of CPU, when calculating distance")
    parser.add_argument("--MEM_weight",
                        type=float,
                        default=1.,
                        help="The weight of Memory, when calculating distance")
    parser.add_argument("--MEM_penalty",
                        type=float,
                        default=100.,
                        help="The weight of Memory, when calculating distance")
    return parser.parse_args()