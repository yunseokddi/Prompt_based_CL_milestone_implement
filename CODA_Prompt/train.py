import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('CODA-Prompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_coda_prompt':
        config_parser = subparser.add_parser('cifar100_coda_prompt', help='Split-CIFAR100 CODA-Prompt configs')
