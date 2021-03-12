import neptune

use_neptune = False

def init_log(args):
    global use_neptune
    if args.neptune:
        with open(args.neptune_path, 'r') as f:
            nep = f.readlines()
            neptune.init(nep[0].strip(), api_token=nep[1].strip())
            neptune.create_experiment(params=vars(args), upload_source_files=['*.py'])
            use_neptune = True


def send_log(key, value):
    global use_neptune
    if use_neptune:
        try:
            neptune.send_metric(key, value)
        except:
            print("Log failed: ", key, value)

def set_log_property(key, value):
    if use_neptune:
        try:
            neptune.set_property(key, value)
        except:
            print("Log property failed: ", key, value)
