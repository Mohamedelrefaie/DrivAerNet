# ============ Function Usage ============
'''
1. 
    world_size = len(gpu_list.split(','))
#! 
    gpu_list                   -> "0, 1, 2"
    gpu_list.split(',')        -> ['0', '1', '2']
    len(gpu_list.split(','))   -> 3

2.
    exp_dir = os.path.join('experiments', args.exp_name)
#!
    Build a path "./experiments/exp_name"

3.
    os.makedirs(exp_dir, exist_ok=True)
#!
    Create the directory "exp_dir" if it doesn't already exist

4. 
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)
#!
   train_and_evaluate(rank, world_size, args)
   # rank :        which GPU this process is using
   # world_size :  total number of GPUS
   # args:         your parsed command-line arguments 
#!
    mp.spawn(...)
    ->
    train_evaluate(rank=0, world_size= , args=args)
    train_evaluate(rank=1, world_size= , args=args)
    train_evaluate(rank=2, world_size= , args=args)
    train_evaluate(rank=3, world_size= , args=args)
    ...
    

'''
