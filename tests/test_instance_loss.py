import torch

from radfoam_model.instance_loss import (
    IGNORE_LABEL,
    MASK_STRIDE,
    instance_contrastive_loss,
    multi_level_instance_loss,
)


def label(view, mask):
    return view * MASK_STRIDE + mask


def test_perfectly_separated_features_have_no_loss():
    # Two masks in one view, each pixel already exactly at its prototype and
    # the prototypes further apart than the margin.
    features = torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
    labels = torch.tensor([label(0, 1), label(0, 1), label(0, 2), label(0, 2)])

    out = instance_contrastive_loss(features, labels, gamma=1.0)

    assert out["positive"].item() == 0.0
    assert out["negative"].item() == 0.0
    assert out["n_prototypes"] == 2


def test_positive_term_penalises_spread_within_a_mask():
    features = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([label(0, 1), label(0, 1)])

    out = instance_contrastive_loss(features, labels)

    # prototype is (1, 0); each pixel is distance 1 away, squared = 1
    assert out["positive"].item() == 1.0
    assert out["negative"].item() == 0.0      # only one prototype, no pairs


def test_negative_term_fires_when_prototypes_are_too_close():
    features = torch.tensor([[0.0, 0.0], [0.2, 0.0]])
    labels = torch.tensor([label(0, 1), label(0, 2)])

    out = instance_contrastive_loss(features, labels, gamma=1.0)

    # distance 0.2, margin 1.0 -> relu(1 - 0.2) = 0.8
    assert abs(out["negative"].item() - 0.8) < 1e-6
    assert out["n_pairs"] == 1


def test_prototypes_in_different_views_are_never_pushed_apart():
    """The whole point of the same-view restriction.

    Two views each see one mask, and the features coincide. A naive
    implementation would push them apart; here there is no valid pair at all,
    because the same object seen twice *should* share a feature.
    """
    features = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    labels = torch.tensor([label(0, 1), label(7, 1)])

    out = instance_contrastive_loss(features, labels, gamma=1.0)

    assert out["n_prototypes"] == 2
    assert out["n_pairs"] == 0
    assert out["negative"].item() == 0.0


def test_pairs_are_counted_within_views_only():
    # view 0 has 3 masks (3 pairs), view 1 has 2 masks (1 pair) -> 4, not C(5,2)=10
    features = torch.randn(5, 4)
    labels = torch.tensor(
        [label(0, 1), label(0, 2), label(0, 3), label(1, 1), label(1, 2)]
    )

    out = instance_contrastive_loss(features, labels)

    assert out["n_pairs"] == 4


def test_ignore_label_is_excluded():
    features = torch.tensor([[0.0, 0.0], [9.0, 9.0], [0.0, 0.0]])
    labels = torch.tensor([label(0, 1), IGNORE_LABEL, label(0, 1)])

    out = instance_contrastive_loss(features, labels)

    # the ignored ray must not drag the prototype toward (9, 9)
    assert out["positive"].item() == 0.0
    assert out["n_prototypes"] == 1


def test_all_ignored_returns_zero_without_error():
    features = torch.randn(4, 8)
    labels = torch.full((4,), IGNORE_LABEL)

    out = instance_contrastive_loss(features, labels)

    assert out["total"].item() == 0.0
    assert out["n_prototypes"] == 0


def test_gradient_flows_to_features():
    features = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([label(0, 1)] * 3 + [label(0, 2)] * 3)

    instance_contrastive_loss(features, labels)["total"].backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_multi_level_sums_over_levels():
    features = torch.randn(4, 3)
    # level 0 splits the rays, level 1 merges them
    level_labels = torch.tensor([
        [label(0, 1), label(0, 1)],
        [label(0, 1), label(0, 1)],
        [label(0, 2), label(0, 1)],
        [label(0, 2), label(0, 1)],
    ])

    out = multi_level_instance_loss(features, level_labels)

    assert "l0_positive" in out and "l1_positive" in out
    assert out["l0_prototypes"] == 2
    assert out["l1_prototypes"] == 1
    assert torch.isfinite(out["total"])


def test_padding_does_not_create_phantom_pairs():
    """Views with unequal mask counts pad to a common width.

    If the padding slots were treated as real prototypes they would sit at the
    origin, be within the margin of everything, and add a large spurious loss.
    """
    features = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    labels = torch.tensor(
        [label(0, 1), label(0, 2), label(0, 3), label(5, 1)]
    )

    out = instance_contrastive_loss(features, labels, gamma=1.0)

    assert out["n_pairs"] == 3          # only view 0's three masks pair up
    assert out["negative"].item() == 0.0  # and they are all far apart
