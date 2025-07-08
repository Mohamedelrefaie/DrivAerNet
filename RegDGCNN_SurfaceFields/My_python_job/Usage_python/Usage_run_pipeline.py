# ======== Hyperparameter ========
1.
    self.k = args['k']
    -> k
    -> Number of nearnest neighbors(for graph edge construction)

2.

# ======== function usage ========

#!-------------------------
    for key, value in env.items():
        print(f"{key} = {value}")

key                 : the name of environment variable
value               : key's value
f"{key} = {value}"  : f-strings(formatted strings)

#!-------------------------
    stages = args.stages.splits(',') if ',' in args.stages else [args.exp_name]
    This is a ternary expression
    stages = <A> if <condition> else <B>

    -> e.g. --stage "preprocess, train"
       -> stages = ['preprocess, train']

    -> e.g. --stage "all"
       -> stages = [all]

#!-------------------------
    results = {}
    -> create an empty directory
    -> e.g. results = {
        "preprocess": True,
        "train":      flase,
        "evaluate":   True
    }
    -> e.g. results.get('train', False)
        -> resluts have train, get True
           whether get False

#!-------------------------
    logging.info
    -> Need explicit declaration



#!-------------------------
def preprocess_data(args):
    try:
        # some code ...
    except Exception as e:
        logging.error(f"Preprocessing failed with error: {e}")
        return False
    -> If any error happens in the try block, Python immediately jumps to except block instead of crashing
    -> The error object is saved as e

