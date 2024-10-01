import numpy as np
from floodlight import XY
from floodlight.models.distance_model import DistanceModel

def generate_random_xy_data(num_frames, num_points, xlim=(0, 108), ylim=(0, 68)):
    """Generate random XY data for testing."""
    data = np.empty((num_frames, num_points * 2))  
    data[:] = np.nan  # Fill with NaNs

    for frame in range(num_frames):
        x_coords = np.random.uniform(xlim[0], xlim[1], num_points)
        y_coords = np.random.uniform(ylim[0], ylim[1], num_points)
        
        data[frame, :num_points] = x_coords  # X coordinates for the current frame
        data[frame, num_points:] = y_coords  # Y coordinates for the current frame

    return XY(data)

def test_distance_model_nearest_mate():
    """Test the DistanceModel for distance to nearest mate."""
    xy1 = generate_random_xy_data(num_frames=3, num_points=11)
    dm = DistanceModel()
    
    dm.fit(xy1)
    dtnm = dm.distance_to_nearest_mate()
    
    assert dtnm is not None
    assert isinstance(dtnm.property, np.ndarray)
    assert dtnm.property.shape[0] == 3

def test_distance_model_team_spread():
    """Test the DistanceModel for team spread calculation."""
    xy1 = generate_random_xy_data(num_frames=3, num_points=11)
    dm = DistanceModel()
    
    dm.fit(xy1)
    spread = dm.team_spread()

    assert spread is not None
    assert isinstance(spread.property, np.ndarray)
    assert spread.property.shape[0] == 3

def test_distance_model_opponents():
    """Test the DistanceModel for distance to nearest opponents."""
    xy1 = generate_random_xy_data(num_frames=3, num_points=11)
    xy2 = generate_random_xy_data(num_frames=3, num_points=11, xlim=(50, 158), ylim=(50, 158))
    
    dm = DistanceModel()
    dm.fit(xy1, xy2)
    
    dtno1, dtno2 = dm.distance_to_nearest_opponents()
    
    assert dtno1 is not None
    assert dtno2 is not None
    assert isinstance(dtno1.property, np.ndarray)
    assert isinstance(dtno2.property, np.ndarray)
    assert dtno1.property.shape[0] == 3
    assert dtno2.property.shape[0] == 3
