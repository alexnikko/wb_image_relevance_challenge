import torch

from src.optimization import step_optimizer


def test_amp_gradients_are_unscaled_before_clipping(monkeypatch):
    events = []
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    class FakeScaler:
        def unscale_(self, _optimizer):
            events.append("unscale")

        def step(self, _optimizer):
            events.append("step")

        def update(self):
            events.append("update")

    def record_clip(_parameters, max_norm):
        assert max_norm == 1.0
        events.append("clip")
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    step_optimizer(model.parameters(), optimizer, max_norm=1.0, scaler=FakeScaler())

    assert events == ["unscale", "clip", "step", "update"]
