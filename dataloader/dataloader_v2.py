from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):

    data = import_module('dataloader.' + 'dataset_v2')

    data_train = getattr(data, 'TrainSet')(args)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)

    data_test = getattr(data, 'TestSet')(args)
    dataloader_test  = DataLoader(data_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    data_eval = getattr(data, 'EvalSet')(args)
    dataloader_eval  = DataLoader(data_eval,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return {'train': dataloader_train, 'test': dataloader_test, 'eval': dataloader_eval}
