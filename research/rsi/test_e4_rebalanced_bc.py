"""E4 fail-closed source-only checks. Never substitute these for Taisei gameplay."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from e4_rebalanced_bc import DATASET_SHA,BASE_BC_SHA,make_weights,EPOCHS,SEED

class E4Tests(unittest.TestCase):
    def test_pinned_source_hashes_and_hyperparameters(self):
        self.assertEqual(DATASET_SHA,"02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d")
        self.assertEqual(BASE_BC_SHA,"87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54")
        self.assertEqual((EPOCHS,SEED),(16,48030))

    def test_class_weights_are_finite_and_normalized(self):
        labels=np.array([0]*100+[1]*10+[12]*8+[17]*2,dtype=np.int64)
        w,counts=make_weights(labels)
        self.assertTrue(np.isfinite(w).all())
        self.assertAlmostEqual(float(np.dot(w,counts)/len(labels)),1,places=5)
        self.assertGreater(w[17],w[0])
        self.assertEqual(int(counts[17]),2)

    def test_reject_invalid_labels(self):
        for labels in ([18,0],[0,-1],[],[True,False],[1.5,0.2]):
            with self.assertRaises(ValueError):
                make_weights(np.asarray(labels))

if __name__=="__main__":
    unittest.main()
