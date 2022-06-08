import deepspeed
import wandb


def get_ds_model(args, colbert):
    config = {
        "train_batch_size": args.bsize,
        "gradient_accumulation_steps": args.accumsteps if "accumsteps" in args else 2,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr if "lr" in args else 3e-06,
                "eps": 1e-8
            }
        },
        "fp16": {
            "enabled": False,
        },
        # this option only available when zero is not activated.
        # "amp": {
        #     "enabled": args.amp,
        # },
        "zero_optimization": {
            # pe-colbert (즉, colbert 계열)에서는 zero를 적용하지 않는 게 throughput 이 높다.
            # [106, 101, 99, ~= 75] = [0, 1, 2, 3]
            # throughput 개선은 물론 mem, cpu, gpu, gpu-mem 리소스의 적절한 콜라보를 위해서는
            # 적절한 딥스피드 설정을 가지고 있어야할 것 같다.
            "stage": 0,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            # offload를 사용하면 GPU에서 메모리 부족이슈는 해결 가능하지만 throughput은 저하될 수 있다.
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            # "offload_param": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            # "sub_group_size": 1e14,
            # "stage3_max_live_parameters": 1e9,
            # "stage3_max_reuse_distance": 1e9,
        },
    }
    if args.rank < 1:
        wandb.config.update({"deepspeed_config": config}, allow_val_change=True)

    deepspeed.init_distributed()

    model, optimizer, _, _ = deepspeed.initialize(model=colbert,
                                                  config_params=config,
                                                  model_parameters=colbert.parameters())
    return model, optimizer
