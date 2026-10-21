"""The exportable SigLIP rewrite must be numerically identical to the tower.

The whole point of the rewrite is that it changes the GRAPH, not the
arithmetic. If the unfolded-patch projection is off by even a small
amount, the exported model's scores would differ from the measured probe
and the shipped detector would not be the one that was evaluated.
"""
import unittest

import numpy as np
import torch
import torch.nn as nn


class _FakeAttention(nn.Module):
    """A minimal stand-in for the pooling head, so the test needs no weights."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, hidden_state):
        probe = self.probe.repeat(hidden_state.shape[0], 1, 1)
        attended = probe + hidden_state.mean(dim=1, keepdim=True)
        residual = attended
        attended = self.layernorm(attended)
        return (residual + self.mlp(attended))[:, 0]


class _FakeEmbeddings(nn.Module):
    def __init__(self, dim: int, patches: int, patch: int) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, padding="valid")
        self.position_embedding = nn.Embedding(patches, dim)
        self.position_ids = torch.arange(patches).unsqueeze(0)
        self.register_buffer("_pos", torch.zeros(patches, dim))

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.position_ids)


class _FakeLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, hidden_states, attention_mask=None):
        return self.lin(hidden_states)


class _EncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(2)])

    def forward(self, inputs_embeds, attention_mask=None):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return _EncoderOutput(hidden)


class _FakeTower(nn.Module):
    def __init__(self, dim: int = 32, patches: int = 64, patch: int = 8) -> None:
        # 64x64 input with an 8px patch -> an 8x8 grid -> 64 patches.
        super().__init__()
        self.embeddings = _FakeEmbeddings(dim, patches, patch)
        self.encoder = _FakeEncoder(dim)
        self.post_layernorm = nn.LayerNorm(dim)
        self.head = _FakeAttention(dim)


class ExportableSiglipProbeTest(unittest.TestCase):
    def _build(self):
        from export.siglip_exportable import ExportableSiglipProbe

        torch.manual_seed(0)
        tower = _FakeTower()
        # Give the position embedding a non-trivial value.
        with torch.no_grad():
            tower.embeddings.position_embedding.weight.copy_(
                torch.randn_like(tower.embeddings.position_embedding.weight)
            )
        probe_weight = torch.randn(1, tower.head.probe.shape[-1])
        probe_bias = torch.zeros(1)
        return tower, ExportableSiglipProbe(tower, probe_weight, probe_bias), probe_weight, probe_bias

    def test_the_rewritten_projection_matches_the_convolution(self):
        """The unfolded-patch Linear must equal Conv2d exactly."""
        tower, model, _, _ = self._build()
        x = torch.rand(2, 3, 64, 64)

        with torch.no_grad():
            reference = tower.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
            rewritten = model.patch_features(x)
        self.assertEqual(reference.shape, rewritten.shape)
        error = float((reference - rewritten).abs().max())
        self.assertLess(error, 1e-5, f"projection mismatch: {error}")

    def test_the_full_forward_matches_the_reference_path(self):
        tower, model, probe_weight, probe_bias = self._build()
        # The exported graph takes NHWC, the layout the app uses.
        nchw = torch.rand(2, 3, 64, 64)
        x = nchw.permute(0, 2, 3, 1)
        feat_mean = torch.zeros(32)
        feat_std = torch.ones(32)
        pix_mean = torch.tensor([0.5, 0.5, 0.5])
        pix_std = torch.tensor([0.5, 0.5, 0.5])
        # The exported model normalises pixels and features; the reference
        # path must do the same or the comparison is meaningless.
        model.pix_mean.copy_(pix_mean)
        model.pix_std.copy_(pix_std)
        model.feat_mean.copy_(feat_mean)
        model.feat_std.copy_(feat_std)

        with torch.no_grad():
            rgb = (nchw - pix_mean.view(1, 3, 1, 1)) / pix_std.view(1, 3, 1, 1)
            reference = tower.embeddings(rgb)
            # The ENCODER is called, not the layers: in transformers 5.x a
            # layer returns a TENSOR, so `layer(...)[0]` silently slices
            # the batch dimension.
            reference = tower.encoder(inputs_embeds=reference, attention_mask=None).last_hidden_state
            reference = tower.post_layernorm(reference)
            reference = tower.head(reference).float()
            reference = reference / (reference.norm(dim=1, keepdim=True) + 1e-12)
            reference = ((reference - feat_mean) / feat_std) @ probe_weight.reshape(-1) + probe_bias.reshape(-1)
            reference = torch.sigmoid(reference).reshape(-1, 1)

            rewritten = model(x)

        self.assertEqual(reference.shape, rewritten.shape)
        error = float((reference - rewritten).abs().max())
        self.assertLess(error, 1e-5, f"forward mismatch: {error}")

    def test_the_graph_contains_no_four_dimensional_convolution(self):
        """This is the property that makes onnx2tf work.

        onnx2tf rewrites 4-D convolutions from NCHW to NHWC and loses the
        channel axis, which is what produced the LayerNorm error
        ("has 768 elements, but got shape (196,)").
        """
        import io

        tower, model, _, _ = self._build()
        buffer = io.BytesIO()
        torch.onnx.export(
            model, torch.rand(1, 64, 64, 3), buffer,
            input_names=["image"], output_names=["probability"],
            opset_version=17, dynamo=False,
        )
        import onnx

        graph = onnx.load_from_string(buffer.getvalue()).graph
        conv_nodes = [
            node for node in graph.node
            if node.op_type == "Conv" and len(node.inputs) and "patch" in node.name.lower()
        ]
        self.assertEqual(
            conv_nodes, [],
            "the patch projection must not appear as a Conv in the exported graph",
        )
        # And no Conv at all.
        all_convs = [node for node in graph.node if node.op_type == "Conv"]
        self.assertEqual(all_convs, [], "the exported graph must contain no Conv nodes")

    def test_a_non_multiple_of_the_patch_size_is_rejected(self):
        tower, model, _, _ = self._build()
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 30, 64))


if __name__ == "__main__":
    unittest.main()
