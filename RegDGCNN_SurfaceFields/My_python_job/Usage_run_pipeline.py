# ======== function usage ========
'''
1.
    for key, value in env.items():
        print(f"{key} = {value}") 

key                 : the name of environment variable
value               : key's value
f"{key} = {value}"  : f-strings(formatted strings)

2. 
    stages = args.stages.splits(',') if ',' in args.stages else [args.exp_name]
#!
    This is a ternary expression
    stages = <A> if <condition> else <B>
#!
    e.g. --stage "preprocess, train"
    -> stages = ['preprocess, train']

    e.g. --stage "all"
    -> stages = [all]

3. 
    results = {}
#!
    create an empty directory
#!
    e.g. results = {
        "preprocess": True, 
        "train":      flase,
        "evaluate":   True
    }

    e.g. results.get('train', False)
    -> resluts have train, get True
       whether get False
4.
    logging.info
#!
    Need explicit declaration
'''


