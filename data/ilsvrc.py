import os
import data.misc as misc
from data.arbitrary import Arbitrary

class ILSVRC(Arbitrary):
    super(ILSVRC, self).__init__()
    
    def load_dataset(self):
        """
        :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
        """
        dataset = {}
        path = os.path.expanduser(self.root)
        classes = os.listdir(path)
        classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
        for i, cls in enumerate(classes):
            if self.verbose:
                print('Loading {}th {} class.'.format(i, cls))
            dataset.update({cls: [_ for _ in os.listdir(os.path.join(path, cls)) if
                                  misc.extension_check(_, self.extensions)]})
        print("Dataset loading is complete.")
        return dataset