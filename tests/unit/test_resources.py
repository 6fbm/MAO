from __future__ import annotations

from mao.orchestration.resources import ResourceMonitor, ResourceSnapshot


def test_resource_monitor_samples_memory() -> None:
    monitor = ResourceMonitor(enable_gpu=False)
    snapshot = monitor.sample()
    assert snapshot.ram_percent is not None and 0 < snapshot.ram_percent <= 100
    assert snapshot.cpu_percent is not None
    assert "RAM" in snapshot.render()
    assert monitor.sample() is snapshot  # cached within the sampling interval
    assert monitor.ram_pressure(1.0)
    assert not monitor.ram_pressure(None)


def test_snapshot_render_with_gpu() -> None:
    snapshot = ResourceSnapshot(cpu_percent=12, ram_percent=50, ram_used_gb=8, ram_total_gb=16, gpu_name="RTX", gpu_used_mb=1200, gpu_total_mb=8192, gpu_util_percent=5)
    assert snapshot.render() == "CPU 12 % · RAM 50 % (8.0/16.0 GB) · GPU RTX 1200/8192 MB (5 %)"
    assert ResourceSnapshot().render() == "no measurements"
