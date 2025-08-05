import numpy as np
import pyvista as pv

def write_pointnet_vtk(points,
                       pressure,
                       fname: str = "pressure_cloud.vtk",
                       binary: bool = True) -> None:
    """
    :param points: (N,3) float array of xyz positions
    :param pressure: (N,) or (N,1) float array per point
    """
    assert points.ndim == 2 and points.shape[1] == 3, \
        f"points should be (N,3), got {points.shape}"
    pressure = pressure.reshape(-1)
    assert pressure.shape[0] == points.shape[0], \
        "pressure must have same N as points"

    # Wrap into a PyVista PolyData
    cloud = pv.PolyData(points)

    # Attach per-point scalar array
    cloud["pressure"] = pressure

    # (Optional) Quick visualization check
    try:
        cloud.plot(render_points_as_spheres=True,
                   scalars="pressure",
                   point_size=5,
                   cmap="viridis")
    except Exception:
        pass

    # Save to .vtk (legacy VTK format)
    cloud.save(fname, binary=binary)
    print(f"Saved {points.shape[0]} points with 'pressure' → {fname}")

