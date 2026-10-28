import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import export_qwen08b_rag_pack as exporter


class TestExportQwen08bRagPack(unittest.TestCase):
    def test_self_test_passes(self):
        rc = exporter.run_self_test()
        self.assertEqual(rc, 0)

    def test_policy_bundle_contract_dimensions(self):
        bundle = exporter.build_default_policy_bundle("Qwen/Qwen3.5-0.8B", hidden_dim=1024)
        self.assertEqual(bundle["schema_version"], "onn-forensic-vlm-policy-v1")
        hc = bundle["head_contract"]
        self.assertEqual(hc["hidden_dim"], 1024)
        self.assertEqual(len(hc["w_class"]), 1024)
        self.assertEqual(len(hc["w_forgery_axes"]), 6)
        self.assertEqual(len(hc["w_counter_axes"]), 6)


if __name__ == "__main__":
    unittest.main()
