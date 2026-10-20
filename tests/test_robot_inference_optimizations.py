from contextlib import nullcontext
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from slot_jepa.predictors.action_mask import (
    ActionConditionalMaskPredictor,
    ObjectScalePredictorBranch,
    ReconstructedFeaturePredictorBranch,
)
from slot_jepa.predictors.components.modules import (
    ACRoPEAttention,
    _build_rope_factors,
    rotate_queries_or_keys,
)


ROOT = Path(__file__).resolve().parents[1]
SERVE_SLOT_PATH = ROOT / "workflows" / "robot" / "serve_slot.py"
SPEC = importlib.util.spec_from_file_location("release_robot_serve_slot", SERVE_SLOT_PATH)
SERVE_SLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVE_SLOT)


def _history_item(index, *, dtype=torch.float16):
    generator = torch.Generator().manual_seed(index)
    logits = torch.randn(1, 4, 9, generator=generator, dtype=dtype)
    probs = torch.softmax(logits.float(), dim=1).to(dtype)
    return {
        "original_index": index,
        "context": {
            "decoded_mask_logits": logits,
            "decoded_mask_probs": probs,
        },
        "state": torch.randn(1, 7, generator=generator, dtype=dtype),
        "extrinsics": torch.randn(1, 6, generator=generator, dtype=dtype),
    }


class CandidateHistoryBatchingTest(unittest.TestCase):
    def test_indexed_pool_is_exactly_equivalent_to_legacy_cat_stack(self):
        items = [_history_item(index) for index in range(5)]
        selected = [
            [items[0], items[1], items[2], items[3]],
            [items[0], items[2], items[2], items[4]],
            [items[1], items[0], items[3], items[4]],
        ]
        context_frames = 3

        legacy_logits = torch.stack(
            [
                torch.cat(
                    [item["context"]["decoded_mask_logits"] for item in candidate],
                    dim=0,
                )
                for candidate in selected
            ],
            dim=0,
        ).float()[:, -context_frames:]
        legacy_probs = torch.stack(
            [
                torch.cat(
                    [item["context"]["decoded_mask_probs"] for item in candidate],
                    dim=0,
                )
                for candidate in selected
            ],
            dim=0,
        ).float()[:, -1]
        legacy_states = torch.stack(
            [torch.cat([item["state"] for item in candidate], dim=0) for candidate in selected],
            dim=0,
        ).float()[:, -context_frames:]
        legacy_extrinsics = torch.stack(
            [
                torch.cat([item["extrinsics"] for item in candidate], dim=0)
                for candidate in selected
            ],
            dim=0,
        ).float()[:, -context_frames:]

        logits, probs, states, extrinsics = SERVE_SLOT._batch_candidate_history_tensors(
            selected,
            context_frames=context_frames,
            view_name="test_view",
        )

        self.assertTrue(torch.equal(logits, legacy_logits))
        self.assertTrue(torch.equal(probs, legacy_probs))
        self.assertTrue(torch.equal(states, legacy_states))
        self.assertTrue(torch.equal(extrinsics, legacy_extrinsics))

    def test_missing_context_validation_still_checks_trimmed_frames(self):
        items = [_history_item(index) for index in range(4)]
        items[0] = dict(items[0], context=None)
        with self.assertRaisesRegex(ValueError, r"history indices \[0\]"):
            SERVE_SLOT._batch_candidate_history_tensors(
                [items],
                context_frames=2,
                view_name="test_view",
            )


class CandidateHistoryMetadataCacheTest(unittest.TestCase):
    @staticmethod
    def _agent():
        agent = object.__new__(SERVE_SLOT.SlotWorldModelAgent)
        agent.memory_selection_mode = "relevance"
        agent.history_len = 3
        agent.gripper_history_bank_by_view = {"test_view": []}
        agent.gripper_position_min = 0.0
        agent.gripper_position_max = 1.0
        agent.gripper_history_pose_distance_weight = 0.2
        agent.history_pose_sampling_temperature = 0.05
        agent.history_pose_uniform_mix = 0.15
        return agent

    @staticmethod
    def _items():
        return [
            {
                "original_index": index,
                "robot_pose6": np.asarray(
                    [0.01 * index, -0.02 * index, 0.03 * index, 0.0, 0.0, 0.01 * index],
                    dtype=np.float32,
                ),
                "gripper_state": float(index) / 10.0,
                "goal_loss": None,
                "source": None,
                "context": {"token": index},
                "state": torch.full((1, 7), float(index)),
                "extrinsics": torch.full((1, 6), float(index)),
            }
            for index in range(6)
        ]

    def test_prebuilt_metadata_preserves_selection_and_debug_without_mutation(self):
        target_pose = np.asarray([0.025, -0.015, 0.04, 0.01, 0.0, 0.02], dtype=np.float32)
        base_items = self._items()

        legacy_agent = self._agent()
        legacy_agent._all_history_items_for_view = lambda _view: [dict(item) for item in base_items]
        np.random.seed(1234)
        legacy_selected, legacy_debug = legacy_agent._select_candidate_history_items(
            "test_view",
            target_pose6_robot=target_pose,
            target_gripper=0.35,
            max_items=3,
            mode="pose_weighted_random",
        )

        cached_agent = self._agent()
        cached_agent._all_history_items_for_view = lambda _view: self.fail(
            "cached selection unexpectedly rematerialized history"
        )
        np.random.seed(1234)
        cached_selected, cached_debug = cached_agent._select_candidate_history_items(
            "test_view",
            target_pose6_robot=target_pose,
            target_gripper=0.35,
            max_items=3,
            mode="pose_weighted_random",
            history_items=base_items,
        )

        self.assertEqual(
            [item["original_index"] for item in cached_selected],
            [item["original_index"] for item in legacy_selected],
        )
        self.assertEqual(cached_debug, legacy_debug)
        self.assertTrue(
            all("candidate_pose_distance" not in item for item in base_items)
        )

    def test_prepared_pose_distance_is_exactly_equal(self):
        agent = self._agent()
        target_pose = np.asarray([0.025, -0.015, 0.04, 0.01, -0.03, 0.02], dtype=np.float32)
        target_rotation_inverse = SERVE_SLOT.Rotation.from_euler(
            "xyz", target_pose[3:6], degrees=False
        ).inv()
        for item in self._items():
            history_pose = item["robot_pose6"]
            history_rotation = SERVE_SLOT.Rotation.from_euler(
                "xyz", history_pose[3:6], degrees=False
            )
            self.assertEqual(
                agent._pose_history_distance(target_pose, history_pose),
                agent._pose_history_distance_prepared(
                    target_pose,
                    target_rotation_inverse,
                    history_pose,
                    history_rotation,
                ),
            )


class TransportLossReuseTest(unittest.TestCase):
    def test_weighted_per_slot_result_matches_full_transport_loss(self):
        agent = object.__new__(SERVE_SLOT.SlotWorldModelAgent)
        agent._ot_cost_cache = {}
        agent.mask_centralize_gamma = 2.0
        agent.unbalanced_ot_eps = 1.0e-8
        agent.unbalanced_ot_rho = 0.2
        agent.sinkhorn_entropy_reg = 0.05
        agent.sinkhorn_iters = 3
        agent.mask_transport_area_normalize = True
        agent.mask_transport_area_normalize_eps = 1.0e-3

        generator = torch.Generator().manual_seed(2026)
        pred = torch.softmax(torch.randn(3, 4, 9, generator=generator), dim=1)
        target = torch.softmax(torch.randn(3, 4, 9, generator=generator), dim=1)
        weights = torch.rand(3, 4, generator=generator).clamp_min(0.05)
        area_normalizer = target.sum(dim=-1).amax(dim=0)

        direct = agent._sinkhorn_mask_transport_loss(
            pred,
            target,
            spatial_size=3,
            chunk_size=5,
            slot_weights=weights,
            area_normalizer=area_normalizer,
        )
        per_slot = agent._sinkhorn_mask_transport_loss_per_slot(
            pred,
            target,
            spatial_size=3,
            chunk_size=5,
            area_normalizer=area_normalizer,
        )
        reused = agent._weighted_slot_loss_mean(per_slot, slot_weights=weights).float()

        self.assertTrue(torch.equal(reused, direct))


class WarpReuseTest(unittest.TestCase):
    def test_predictor_and_caller_receive_one_shared_warp_result(self):
        class Predictor:
            use_warp_concat = True

            def __init__(self):
                self.external_warped = None

            def __call__(self, **kwargs):
                self.external_warped = kwargs["external_warped_mask_probs"]
                return torch.ones(kwargs["next_states"].shape[0], 1, 4, 9)

        agent = object.__new__(SERVE_SLOT.SlotWorldModelAgent)
        agent.use_extrinsics = False
        agent.predictor = Predictor()
        agent._autocast_context = lambda: nullcontext()
        calls = []
        expected_warp = torch.randn(3, 1, 2, 4, 9)

        def build_warp(*args, **kwargs):
            calls.append((args, kwargs))
            return expected_warp

        agent._build_warped_masks_from_cache = build_warp
        cache = {
            "context_masks": torch.randn(1, 2, 4, 9),
            "obj_latents": torch.randn(1, 4, 8),
            "poses_view": torch.randn(1, 2, 7),
            "extr_view": torch.randn(1, 2, 6),
        }
        next_states = torch.randn(3, 1, 7)

        logits, returned_warp = SERVE_SLOT.SlotWorldModelAgent._predict_from_cache(
            agent,
            cache,
            next_states=next_states,
            use_copied_warp=True,
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(returned_warp, expected_warp)
        self.assertIs(agent.predictor.external_warped, expected_warp)
        self.assertEqual(tuple(logits.shape), (3, 1, 4, 9))

    def test_warp_softmax_is_forwarded_to_predictor_without_recomputation(self):
        class Predictor:
            use_warp_concat = True

        agent = object.__new__(SERVE_SLOT.SlotWorldModelAgent)
        agent.predictor = Predictor()
        agent.use_extrinsics = False
        masks = torch.randn(1, 2, 4, 9)
        cache = {"context_masks": masks}

        warped, mask_probs, obj_latent_low = (
            SERVE_SLOT.SlotWorldModelAgent._build_warped_masks_from_cache(
                agent,
                cache,
                next_states=torch.randn(3, 1, 7),
                use_copied_warp=True,
                return_precomputed=True,
            )
        )

        self.assertIsNone(obj_latent_low)
        self.assertTrue(torch.equal(mask_probs, torch.softmax(masks.expand(3, -1, -1, -1), dim=2)))
        self.assertTrue(
            torch.equal(
                warped,
                mask_probs.unsqueeze(1).expand(-1, 1, -1, -1, -1),
            )
        )


class PredictorInputReuseTest(unittest.TestCase):
    @staticmethod
    def _recon_branch(use_warp_concat):
        return ReconstructedFeaturePredictorBranch(
            embed_dim=4,
            feature_dim=4,
            predictor_embed_dim=4,
            depth=0,
            num_heads=1,
            spatial_size=2,
            input_spatial_size=2,
            num_objects=2,
            state_dim=3,
            use_extrinsics=False,
            concat_mask_input=True,
            use_warp_concat=use_warp_concat,
        ).eval()

    @staticmethod
    def _recon_inputs():
        generator = torch.Generator().manual_seed(17)
        return {
            "obj_latent": torch.randn(1, 2, 4, generator=generator),
            "mask_probs": torch.softmax(
                torch.randn(1, 2, 2, 4, generator=generator), dim=2
            ),
            "states": torch.randn(1, 2, 3, generator=generator),
            "next_states": torch.randn(1, 1, 3, generator=generator),
        }

    def test_identical_source_target_tokens_match_recomputed_path_exactly(self):
        branch = self._recon_branch(use_warp_concat=False)
        inputs = self._recon_inputs()
        with torch.no_grad():
            shared = branch(**inputs)
            recomputed = branch(
                **inputs,
                target_mask_probs=inputs["mask_probs"].clone(),
            )
        self.assertTrue(torch.equal(shared, recomputed))

    def test_warped_target_reuses_only_equal_base_features(self):
        branch = self._recon_branch(use_warp_concat=True)
        inputs = self._recon_inputs()
        obj_latent_low = branch.obj_latent_reduce(inputs["obj_latent"])
        source_warp = torch.rand_like(inputs["mask_probs"])
        target_warp = torch.rand_like(inputs["mask_probs"])
        _, base_parts = branch._build_input(
            inputs["mask_probs"],
            source_warp,
            obj_latent_low,
            return_base_parts=True,
        )
        reused = branch._build_input(
            inputs["mask_probs"],
            target_warp,
            obj_latent_low,
            precomputed_base_parts=base_parts,
        )
        recomputed = branch._build_input(
            inputs["mask_probs"].clone(),
            target_warp,
            obj_latent_low,
        )
        self.assertTrue(torch.equal(reused, recomputed))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_full_bfloat16_predictor_reuse_matches_recomputed_path(self):
        device = torch.device("cuda")
        predictor = ActionConditionalMaskPredictor(
            embed_dim=8,
            feature_dim=8,
            predictor_embed_dim=48,
            depth=1,
            num_heads=3,
            num_context_frames=2,
            spatial_size=2,
            num_objects=2,
            state_dim=3,
            use_extrinsics=False,
            use_warp_concat=True,
            object_branch_scales=[
                {
                    "spatial_size": 2,
                    "dim": 48,
                    "depth": 1,
                    "num_heads": 3,
                }
            ],
        ).to(device).eval()
        generator = torch.Generator(device=device).manual_seed(31)
        obj_latent = torch.randn(1, 2, 8, generator=generator, device=device)
        context_masks = torch.randn(
            1, 2, 2, 4, generator=generator, device=device
        )
        states = torch.randn(1, 2, 3, generator=generator, device=device)
        next_states = torch.randn(1, 1, 3, generator=generator, device=device)
        mask_probs = torch.softmax(context_masks, dim=2)
        copied_warp = mask_probs.unsqueeze(1).contiguous()

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            recomputed = predictor(
                obj_latent,
                context_masks,
                states,
                next_states,
                external_warped_mask_probs=copied_warp,
            )
            reused = predictor(
                obj_latent,
                context_masks,
                states,
                next_states,
                external_warped_mask_probs=copied_warp,
                external_warp_matches_context=True,
                precomputed_mask_probs=mask_probs,
            )
        self.assertTrue(torch.equal(reused, recomputed))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_same_gpu_stream_parallel_outputs_match_sequential_outputs(self):
        device = torch.device("cuda")
        predictor = ActionConditionalMaskPredictor(
            embed_dim=8,
            feature_dim=8,
            predictor_embed_dim=48,
            depth=1,
            num_heads=3,
            num_context_frames=2,
            spatial_size=2,
            num_objects=2,
            state_dim=3,
            use_extrinsics=False,
            use_warp_concat=True,
            object_branch_scales=[
                {
                    "spatial_size": 2,
                    "dim": 48,
                    "depth": 1,
                    "num_heads": 3,
                }
            ],
        ).to(device).eval()
        generator = torch.Generator(device=device).manual_seed(37)

        def make_inputs():
            context_masks = torch.randn(
                2, 2, 2, 4, generator=generator, device=device
            )
            mask_probs = torch.softmax(context_masks, dim=2)
            return {
                "obj_latent": torch.randn(
                    2, 2, 8, generator=generator, device=device
                ),
                "context_masks": context_masks,
                "states": torch.randn(
                    2, 2, 3, generator=generator, device=device
                ),
                "next_states": torch.randn(
                    2, 1, 3, generator=generator, device=device
                ),
                "external_warped_mask_probs": (
                    mask_probs.unsqueeze(1).contiguous()
                ),
                "external_warp_matches_context": True,
                "precomputed_mask_probs": mask_probs,
            }

        first = make_inputs()
        second = make_inputs()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            expected_first = predictor(**first)
            expected_second = predictor(**second)
        torch.cuda.synchronize(device)

        streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        default_stream = torch.cuda.current_stream(device=device)
        outputs = []
        with torch.no_grad():
            for stream, inputs in zip(streams, (first, second)):
                stream.wait_stream(default_stream)
                with torch.cuda.stream(stream), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    outputs.append(predictor(**inputs))
            for stream in streams:
                default_stream.wait_stream(stream)
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(outputs[0], expected_first))
        self.assertTrue(torch.equal(outputs[1], expected_second))

    def test_identical_copied_warp_skips_target_embedding(self):
        branch = self._recon_branch(use_warp_concat=True)
        inputs = self._recon_inputs()
        copied_warp = inputs["mask_probs"]
        calls = []
        hook = branch.predictor_embed.register_forward_hook(
            lambda *_args: calls.append(None)
        )
        try:
            with torch.no_grad():
                branch(
                    **inputs,
                    warped_mask_probs=copied_warp,
                    target_warped_mask_probs=copied_warp,
                    query_source_warped_mask_probs=inputs["mask_probs"][:, -1:],
                )
        finally:
            hook.remove()
        # Source history, query source and learned query target. The identical
        # history target no longer invokes predictor_embed a fourth time.
        self.assertEqual(len(calls), 3)

    def test_object_scale_base_feature_reuse_is_exact(self):
        branch = ObjectScalePredictorBranch(
            input_spatial_size=2,
            spatial_size=2,
            embed_dim=4,
            branch_dim=4,
            depth=0,
            num_heads=1,
            state_dim=3,
            use_extrinsics=False,
            use_warp_concat=True,
            output_spatial_size=2,
        ).eval()
        inputs = self._recon_inputs()
        obj_latent_low = branch.obj_latent_reduce(inputs["obj_latent"])
        source_warp = torch.rand_like(inputs["mask_probs"])
        target_warp = torch.rand_like(inputs["mask_probs"])
        _, base_features = branch._build_tokens(
            inputs["mask_probs"],
            source_warp,
            obj_latent_low,
            return_base_features=True,
        )
        reused = branch._build_tokens(
            inputs["mask_probs"],
            target_warp,
            obj_latent_low,
            precomputed_base_features=base_features,
        )
        recomputed = branch._build_tokens(
            inputs["mask_probs"].clone(),
            target_warp,
            obj_latent_low,
        )
        self.assertTrue(torch.equal(reused, recomputed))


class RoPECacheTest(unittest.TestCase):
    @staticmethod
    def _legacy_rotate(x, pos):
        dim = x.shape[-1]
        omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        freq = torch.einsum("..., f -> ... f", pos, omega)
        emb_sin = freq.sin().squeeze(-1).repeat(1, 1, 1, 2)
        emb_cos = freq.cos().squeeze(-1).repeat(1, 1, 1, 2)
        paired = x.unflatten(-1, (-1, 2))
        first, second = paired.unbind(dim=-1)
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        return (x * emb_cos) + (rotated * emb_sin)

    def test_cached_factors_are_bit_exact_with_legacy_rotation(self):
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(2, 3, 5, 6, generator=generator)
        positions = torch.arange(5)
        factors = _build_rope_factors(6, x.dtype, x.device, positions)
        cached = rotate_queries_or_keys(x, positions, rope_factors=factors)
        legacy = self._legacy_rotate(x, positions)
        self.assertTrue(torch.equal(cached, legacy))

    def test_position_and_factor_tensors_are_reused(self):
        attention = ACRoPEAttention(dim=12, num_heads=3, grid_size=4)
        key1, positions1 = attention._cached_positions(3, 2, 2, torch.device("cpu"))
        key2, positions2 = attention._cached_positions(3, 2, 2, torch.device("cpu"))
        self.assertEqual(key1, key2)
        self.assertTrue(all(a is b for a, b in zip(positions1, positions2)))

        x = torch.randn(1, 3, 12, 4)
        first = attention._cached_rotate(x, positions1[0], key1, "patch_depth")
        factor_count = len(attention._rope_factor_cache)
        second = attention._cached_rotate(x, positions1[0], key1, "patch_depth")
        self.assertEqual(len(attention._rope_factor_cache), factor_count)
        self.assertTrue(torch.equal(first, second))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_bfloat16_cuda_cache_is_bit_exact(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(29)
        x = torch.randn(
            2,
            3,
            8,
            6,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        positions = torch.arange(8, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            factors = _build_rope_factors(6, x.dtype, device, positions)
            cached = rotate_queries_or_keys(
                x, positions, rope_factors=factors
            )
            legacy = self._legacy_rotate(x, positions)
        self.assertTrue(torch.equal(cached, legacy))


if __name__ == "__main__":
    unittest.main()
