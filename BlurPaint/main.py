import datetime, os, argparse, glob
import sys
import torch
import pytorch_lightning as pl
import torch.multiprocessing

from packaging import version
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from utils import instantiate_from_config, get_parser
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from utils.loggers import CUDACallback, SetupCallback, ImageLogger


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # torch.set_float32_matmul_precision('highest')
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = get_parser()

    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    opt.train = True

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        # opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        # elif opt.base:
        #     cfg_fname = os.path.split(opt.base[0])[-1]
        #     cfg_name = os.path.splitext(cfg_fname)[0]
        #     name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    logdir = os.path.join(sys.path[0], logdir)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = OmegaConf.load("confs/denoise-network.yaml")
        # cli = OmegaConf.from_dotlist(unknown)
        #
        config = OmegaConf.merge(configs)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # default to ddp
        trainer_config["accelerator"] = "gpu"
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        gpuinfo = opt.device
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
        gpu_nums = len(opt.device.split(','))
        trainer_config["device"] = gpu_nums

        trainer_config["max_steps"] = configs["model"]["max_steps"]

        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs

        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:02}-{step:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = -1
            default_modelckpt_cfg["params"]["every_n_train_steps"] = 20000

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        # if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        #     print(
        #         'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        #     default_metrics_over_trainsteps_ckpt_dict = {
        #         'metrics_over_trainsteps_checkpoint':
        #             {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
        #              'params': {
        #                  "dirpath": os.path.join(sys.path[0], 'trainstep_checkpoints'),
        #                  "filename": "{epoch:06}-{step:09}",
        #                  "verbose": True,
        #                  'save_top_k': -1,
        #                  'every_n_train_steps': 1000,
        #                  'save_weights_only': True
        #              }
        #              }
        #     }
        #     default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["log_every_n_steps"] = 200
        checkpoint_callback = ModelCheckpoint(**default_modelckpt_cfg["params"])

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs, callbacks=checkpoint_callback)

        trainer.logdir = logdir  ###

        data_config = config.data
        data_config["params"]["train"] = {}
        data_config["params"]["train"]["img_dir"] = opt.input_path
        data_config["params"]["train"]["mask_dir"] = opt.mask_path

        # data
        data = instantiate_from_config(data_config)
        # # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # # calling these ourselves should not be necessary but it is.
        # # lightning still takes care of proper multiprocessing though
        # data.prepare_data()
        data.setup()

        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate

        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        # if not cpu:
        #     ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
        # else:
        #     ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = base_lr
            # print(
            #     "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            #         model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
            print(
                "Setting learning rate to {:.2e}".format(model.learning_rate))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGTERM, melk)
        signal.signal(signal.SIGTERM, divein)

        # run
        if opt.train:
            try:
                ckpt_path = os.path.join(sys.path[0], config.model.ckpt_path)
                if ckpt_path is not None and os.path.isfile(ckpt_path):
                    print("load prev checkpoint")
                    trainer.fit(model, data, ckpt_path=ckpt_path)

                else:
                    trainer.fit(model, data)
            except Exception:
                melk()
                raise

        # if not opt.no_test and not trainer.interrupted:
        #     trainer.test(model)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())




