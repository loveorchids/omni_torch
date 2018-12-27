from easydict import EasyDict as edict

BaseOptions=edict({
    "code_name": "",
    "cover_exist":True,
    "create_path":True,

    "gpu_id":"0",
    "epoch_num":2000,
    "deterministic_train": True,
    "seed": 88,

    "batch_size": 8,
    "learning_rate":1e-4,
    "weight_decay":1e-4,
    "batch_norm":True,
    "finetune":False,


})