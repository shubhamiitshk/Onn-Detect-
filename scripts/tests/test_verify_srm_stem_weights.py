import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class VerifySrmStemWeightsTest(unittest.TestCase):
    def test_srm_stem_weights_verification_passes(self):
        from scripts.verify_srm_stem_weights import verify_srm_stem
        self.assertTrue(verify_srm_stem())

if __name__ == "__main__":
    unittest.main()
