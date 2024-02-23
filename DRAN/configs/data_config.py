import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# fmt: off
def build_parser():
    parser = argparse.ArgumentParser(description='Dataset Config', add_help=False)

    ### data
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')
    parser.add_argument('--crop_file', type=str, default='./crop')
    parser.add_argument('--extra_data', type=str, default=None)  
    
    #args = parser.parse_args()

    return parser
# fmt: on


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


