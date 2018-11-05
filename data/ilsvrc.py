import os
import data.misc as misc
from data.arbitrary import Arbitrary
import data

ALLOW_WARNING = data.ALLOW_WARNING

# Todo: Not Tested Yet

class ILSVRC(Arbitrary):
    def __init__(self, args, sources, modes, load_funcs, dig_level, loader_ops=None, **options):
        super().__init__(args, sources, modes, load_funcs, dig_level, loader_ops, options)
        
    def prepare(self):
        self.dataset = self.load_dataset()
    
    def load_dataset(self):
        """
        :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
        """
        dataset = {}
        path = os.path.expanduser(self.args.path)
        classes = os.listdir(path)
        classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
        for i, cls in enumerate(classes):
            if self.verbose:
                print('Loading {}th {} class.'.format(i, cls))
            dataset.update({cls: [_ for _ in os.listdir(os.path.join(path, cls)) if
                                  misc.extension_check(_, self.args.extensions)]})
        print("Dataset loading is complete.")
        return dataset