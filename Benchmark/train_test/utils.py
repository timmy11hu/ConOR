import json

def get_ckpt_dict(config):
    if config['data']['name'] == "Kitti":
        file_name = "train_test/ckpts_pth_kt.json"
    elif config['data']['name'] == "NYUv2":
        file_name = "train_test/ckpts_pth_nyu.json"
    else:
        raise NotImplementedError
    with open(file_name) as f:
        ckpts_pth = json.load(f)

    return ckpts_pth