import unittest
import torch
from models import build_model
from data import load_data, make_windows


class ModelTests(unittest.TestCase):
    def test_shapes_and_causality(self):
        torch.manual_seed(42)
        tokens=torch.randint(0,20,(2,12))
        changed=tokens.clone()
        changed[:,7:]=(changed[:,7:]+1)%20
        for kind in ['lstm','transformer']:
            model=build_model(kind,20,32).eval()
            with torch.no_grad():
                original=model(tokens)
                other=model(changed)
            self.assertEqual(original.shape,(2,12,20))
            self.assertTrue(torch.allclose(original[:,:7],other[:,:7],atol=1e-6))

    def test_song_split_and_target_shift(self):
        train,valid,chars,info=load_data()
        self.assertFalse(set(info['train_song_hashes'])&set(info['validation_song_hashes']))
        windows=make_windows(train,64)
        x,y=windows[:,:-1],windows[:,1:]
        self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))
        self.assertEqual(chars[0],'<UNK>')
        self.assertGreater(len(valid),0)


if __name__=='__main__': unittest.main(verbosity=2)
